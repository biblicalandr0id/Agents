from genetics import create_genetic_core, GeneticCore
from embryo_namer import EmbryoNamer
from embryo_generator import EmbryoGenerator
from neural_networks import NeuralAdaptiveNetwork
from executor import AdaptiveExecutor
from diagnostics import NeuralDiagnostics
from agent_environment import EnhancedAdaptiveEnvironment, ResourceType, Resource, AgentAction, AdaptiveAgent
import torch
import numpy as np
import json

if __name__ == "__main__":
    genetic_core = create_genetic_core(seed=42)
    embryo_namer = EmbryoNamer()
    generator = EmbryoGenerator(genetic_core, embryo_namer)
    embryo_file = generator.generate_embryo_file()
    size = (100, 100)
    complexity = 1.0
    environment = EnhancedAdaptiveEnvironment(size, complexity)

    input_size = 7
    hidden_sizes = [64, 64]
    output_size = len(AgentAction)
    neural_net = NeuralAdaptiveNetwork(input_size, hidden_sizes[0], output_size)

    diagnostics = NeuralDiagnostics(neural_net)

    agent_position = (50, 50)
    agent = AdaptiveAgent(genetic_core, neural_net, agent_position)
    environment.agents.append(agent)
    executor = AdaptiveExecutor(neural_net)

    env_state = environment.current_state
    env_state.resources = [
        Resource(type=ResourceType.ENERGY, quantity=100, position=(20, 20), complexity=0.2),
        Resource(type=ResourceType.MATERIALS, quantity=50, position=(70, 70), complexity=0.8)
    ]
    env_state.threats = [(80, 80)]

    neural_net.eval() # <-- THIS IS THE LINE ADDED - Set network to evaluation mode - BEFORE the loop

    for step in range(2):
        env_state = environment.current_state
        action, params = agent.decide_action(env_state)
        result = agent.execute_action(action, params)
        executor.record_validation_loss(result.reward)
        loss, outputs, importance = executor.execute(
            inputs=torch.tensor(agent.perceive_environment(env_state)).float().reshape(1, -1),
            targets=torch.tensor(np.zeros(len(AgentAction))).float().reshape(1, -1),
            context=torch.tensor([[0.0]]),
            diagnostics = diagnostics
        )
        diagnostics = agent.neural_diagnostics.monitor_network_health(
            inputs = torch.tensor(agent.perceive_environment(env_state)).float().reshape(1,-1),
            targets = torch.tensor(np.zeros(len(AgentAction))).float().reshape(1,-1),
            context=torch.tensor([[0.0]]),
            epoch=env_state.time_step
        )

        print(json.dumps(diagnostics, indent=2))
        print(f"Step: {step+1}, Action: {action}, Reward: {result.reward:.4f}, Energy: {agent.energy:.2f}")
        environment.step([agent]) # Step the environment forward, passing in list of agents

    print("\nAgent Status after 2 steps:", agent.get_status())