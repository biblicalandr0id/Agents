import numpy as np
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim

class ImprovedAdaptiveNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.experience_buffer = []
        self.max_buffer_size = 1000

    def forward(self, x: torch.Tensor, genetic_modifiers: Dict[str, float]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        activations = [x]

        x = x * genetic_modifiers['sensor_sensitivity']

        for i, layer in enumerate(self.layers[:-1]):
            z = layer(activations[-1])
            z = z * genetic_modifiers['processing_speed']

            a = nn.LeakyReLU(0.01)(z)
            activations.append(a)

        z = self.layers[-1](activations[-1])
        output = torch.softmax(z, dim=-1)
        activations.append(output)

        return output, activations

    def backward(self, x: torch.Tensor, y: torch.Tensor, activations: List[torch.Tensor],
                learning_rate: float, plasticity: float) -> None:
        m = x.shape[0]
        criterion = nn.MSELoss()
        output = activations[-1]
        loss = criterion(output, y)

        self.zero_grad()
        loss.backward()

        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if layer.weight.grad is not None:
                    dW = layer.weight.grad
                    layer.weight -= dW * learning_rate * plasticity

                if layer.bias.grad is not None:
                    db = layer.bias.grad
                    layer.bias -= db * learning_rate * plasticity

class NeuralAdaptiveNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=2, memory_type='lstm'):
        super().__init__()
        self.state_manager = AdaptiveStateManager(
            hidden_size, hidden_size, memory_type=memory_type
        )

        layers = []
        current_size = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size

        layers.append(nn.Linear(current_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x, context):
        batch_size = x.shape[0]
        # Ensure context is properly shaped
        if not isinstance(context, torch.Tensor):
            context = torch.tensor([[context]], dtype=torch.float32)
        if context.dim() == 1:
            context = context.unsqueeze(0)
        if context.shape[0] != batch_size:
            context = context.expand(batch_size, -1)
            
        x = self.network[:-1](x)
        adaptive_state, importance = self.state_manager(x, context)
        output = self.network[-1](adaptive_state)
        return output, importance

class AdaptiveStateManager(nn.Module):
    def __init__(self, input_dim, hidden_dim, adaptive_rate=0.01,
                 memory_layers=2, memory_type='lstm'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adaptive_rate = adaptive_rate
        self.memory_type = memory_type

        if memory_type == 'lstm':
            self.memory_cells = nn.ModuleList([
                nn.LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(memory_layers)
            ])
        else:
            self.memory_cells = nn.ModuleList([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(memory_layers)
            ])

        self.compression_gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )

        self.importance_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )

        self.register_buffer('state_importance', torch.ones(1, hidden_dim))
        self.register_buffer('memory_allocation', torch.ones(1, memory_layers))  # Changed to 2D tensor

    def forward(self, current_state, context):
        batch_size = current_state.shape[0]
        
        # Initialize or update LSTM states based on batch size
        if self.memory_type == 'lstm':
            if not hasattr(self, 'h_state') or self.h_state[0].shape[0] != batch_size:
                self.h_state = [torch.zeros(batch_size, self.hidden_dim, device=current_state.device)
                               for _ in range(len(self.memory_cells))]
                self.c_state = [torch.zeros(batch_size, self.hidden_dim, device=current_state.device)
                               for _ in range(len(self.memory_cells))]

        processed_state = current_state

        if self.memory_type == 'lstm':
            for i, cell in enumerate(self.memory_cells):
                self.h_state[i], self.c_state[i] = cell(processed_state, 
                    (self.h_state[i], self.c_state[i]))
                processed_state = self.h_state[i]
        else:
            for cell in self.memory_cells:
                processed_state = torch.relu(cell(processed_state))

        # Ensure context is properly shaped for concatenation
        if context.dim() == 2 and context.shape[1] != 1:
            context = context[:, :1]
            
        compression_signal = self.compression_gate(
            torch.cat([processed_state, context], dim=-1)
        )
        compressed_state = processed_state * compression_signal

        importance_signal = self.importance_generator(compressed_state)

        with torch.no_grad():
            # Expand state_importance to match batch size
            expanded_importance = self.state_importance.expand(batch_size, -1)
            self.state_importance = expanded_importance + self.adaptive_rate * (
                torch.abs(importance_signal) - expanded_importance
            )

            memory_allocation_update = torch.abs(importance_signal.mean(dim=-1))
            # Reshape memory_allocation_update to match memory_allocation shape
            memory_allocation_update = memory_allocation_update.mean().view(1, 1).expand(1, len(self.memory_cells))
            self.memory_allocation += self.adaptive_rate * (
                memory_allocation_update - self.memory_allocation
            )

        return compressed_state, self.state_importance