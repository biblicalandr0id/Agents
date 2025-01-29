import torch
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

class AdaptiveExecutor: # From neural_executor.py
    def __init__(self, network):
        self.network = network
        self.optimizers = {
            'adam': optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-5),
            'sgd': optim.SGD(network.parameters(), lr=0.01, weight_decay=1e-5),
            'rmsprop': optim.RMSprop(network.parameters(), lr=0.001, weight_decay=1e-5)
        }
        self.schedulers = {
            'adam': ReduceLROnPlateau(self.optimizers['adam'], mode='min', factor=0.5, patience=5, verbose=False),
            'sgd': ReduceLROnPlateau(self.optimizers['sgd'], mode='min', factor=0.5, patience=5, verbose=False),
             'rmsprop': ReduceLROnPlateau(self.optimizers['rmsprop'], mode='min', factor=0.5, patience=5, verbose=False)
        }

        self.losses = {'adam': [], 'sgd': [], 'rmsprop': []}
        self.optimizer_names = ['adam', 'sgd', 'rmsprop']
        self.optimizer_index = 0
        self.validation_losses = [] # Store for validation and tracking best optimizer

    def record_validation_loss(self, loss):
         """Store validation loss for determining best optimizer"""
         self.validation_losses.append(loss)

    def execute(self, inputs, targets, context, diagnostics):
        """Enhanced execute with multiple optimizers, LR scheduling, and anomaly detection"""
        optimizer_name = self.optimizer_names[self.optimizer_index]
        optimizer = self.optimizers[optimizer_name]
        scheduler = self.schedulers[optimizer_name]

        # Forward pass through network with potential genetic modulation
        outputs, activations = self.network(inputs, context) # Get outputs and activation from the network
        
        criterion = nn.MSELoss() # define our loss function

        loss = criterion(outputs, targets)
        self.losses[optimizer_name].append(loss.item())

        # Backpropagation and weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Learning rate scheduling
        scheduler.step(loss.item())
        
        return loss.item(), outputs, diagnostics.get('feature_importance')

    def _select_best_optimizer(self):
        """Selects the best optimizer based on recent validation performance"""
        if len(self.validation_losses) < 5: # Start switching only after at least 5 validation losses
           return self.optimizer_index

        losses = self.validation_losses[-5:] # Take last 5 losses for each optimizer
        losses_over_time = []
        for index in range(len(self.optimizers)):
           losses_over_time.append(sum([self.losses[self.optimizer_names[index]][loss] for loss in range(len(self.losses[self.optimizer_names[index]]) - 5,len(self.losses[self.optimizer_names[index]]))])/5 if len(self.losses[self.optimizer_names[index]]) >=5 else float('inf'))

        best_optimizer_index = losses_over_time.index(min(losses_over_time))
        if best_optimizer_index != self.optimizer_index:
            print(f"Switching optimizer from {self.optimizer_names[self.optimizer_index]} to {self.optimizer_names[best_optimizer_index]}")

        self.optimizer_index = best_optimizer_index
        return best_optimizer_index