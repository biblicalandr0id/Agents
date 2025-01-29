from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import uuid
import numpy as np
import math
import random
import torch
from genetics import GeneticCore
from neural_networks import NeuralAdaptiveNetwork
from executor import AdaptiveExecutor
from diagnostics import NeuralDiagnostics
from augmentation import AdaptiveDataAugmenter
from embryo_namer import EmbryoNamer
import perlin
from adaptive_environment import AdaptiveEnvironment, EnvironmentalState, Resource, ResourceType
from functools import partial

@dataclass
class Resource:
    type: ResourceType
    quantity: float
    position: Tuple[int, int]
    complexity: float
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class EnvironmentalState:
    resources: List[Resource]
    threats: List[Tuple[int, int]]
    time_step: int
    complexity_level: float
    agents: List['AdaptiveAgent'] = field(default_factory=list)

@dataclass
class ActionResult:
    success: bool
    reward: float
    energy_cost: float
    new_state: Optional[Dict]

class ActionVector:
    def __init__(self, selection: torch.Tensor, parameters: torch.Tensor):
      self.selection = selection
      self.parameters = parameters

class ActionDecoder:
    def __init__(self, hidden_size=128):
        self.selection_size = 32  # Base action type encoding
        self.parameter_size = 96  # Action parameters encoding

        # Total hidden_size = selection_size + parameter_size
        self.hidden_size = hidden_size
        # Dictionary mapping action names to their prototype vectors
        self.action_prototypes = {}  
        # The actual functions for each action
        self.action_methods = {}

    def add_action(self, name: str, prototype_vector: torch.Tensor, method):
        self.action_prototypes[name] = prototype_vector
        self.action_methods[name] = method
    def decode_selection(self, selection_vector: torch.Tensor) -> tuple[str, float]:
        # Find closest prototype using cosine similarity
        best_similarity = -1
        selected_action = None

        for name, prototype in self.action_prototypes.items():
            similarity = F.cosine_similarity(
                selection_vector.unsqueeze(0), 
                prototype.unsqueeze(0)
            )

            if similarity > best_similarity:
                best_similarity = similarity
                selected_action = name
        if selected_action is None:
          return None, 0
        return selected_action, best_similarity.item()

class AdaptiveAgent:
    def __init__(self, genetic_core, neural_net, position: Tuple[int, int]):
        self.genetic_core = genetic_core
        self.neural_net = neural_net
        self.position = position

        self.energy = 100.0
        self.resources = {rt: 0.0 for rt in ResourceType}
        self.knowledge_base = {}

        self.total_resources_gathered = 0.0
        self.successful_interactions = 0
        self.survival_time = 0
        self.efficiency_score = 0.0
        self.name = EmbryoNamer().generate_random_name()
        self.data_augmenter = AdaptiveDataAugmenter()
        self.neural_diagnostics = NeuralDiagnostics(neural_net)
        self.actions = {
            "move": self._process_movement,
            "gather": self._process_gathering,
            "process": self._process_resources,
            "share": self._process_sharing,
            "defend": self._process_defense,
            "execute_tool": self._process_tool_execution
        }
        self.action_decoder = ActionDecoder()
        for name, method in self.actions.items():
          self.action_decoder.add_action(name, torch.randn(32), method)

    def augment_perception(self, inputs, context = None):
        return self.data_augmenter.augment(inputs, context)

    def perceive_environment(self, env_state: EnvironmentalState) -> np.ndarray:
        sensor_sensitivity = self.genetic_core.physical_genetics.sensor_sensitivity

        inputs = []

        for resource in env_state.resources:
            distance = self._calculate_distance(resource.position)
            detection_threshold = 10.0 / sensor_sensitivity

            if distance <= detection_threshold:
                clarity = 1.0 - (distance / detection_threshold)
                inputs.extend([
                    1.0,
                    self._normalize_distance(distance, detection_threshold),
                    self._normalize_quantity(resource.quantity),
                    self._normalize_complexity(resource.complexity)
                ])
        if not inputs:
            inputs.extend([0.0] * 4)

        threat_sensitivity = self.genetic_core.heart_genetics.security_sensitivity
        threat_inputs = []
        for threat_pos in env_state.threats:
            distance = self._calculate_distance(threat_pos)
            threat_detection_threshold = 15.0 * threat_sensitivity
            if distance <= threat_detection_threshold:
                threat_inputs.extend([
                    1.0,
                    self._normalize_distance(distance, threat_detection_threshold)
                ])
        if not threat_inputs:
            threat_inputs.extend([0.0] * 2)

        internal_inputs = [
            self._normalize_energy(self.energy),
        ]
        augmented_inputs = self.augment_perception(torch.tensor(inputs + threat_inputs + internal_inputs).float())

        return augmented_inputs.numpy()

    def decide_action(self, env_state: EnvironmentalState) -> Tuple[str, Dict]:
        """Determine next action using neural network and genetic traits"""
        # Get environmental inputs
        sensor_data = self.perceive_environment(env_state)

        # Genetic Modifiers for Neural Network
        genetic_modifiers = {
            'processing_speed': self.genetic_core.brain_genetics.processing_speed,
            'sensor_sensitivity': self.genetic_core.physical_genetics.sensor_sensitivity
        }

        # Process using neural network, modulated by genetic traits
        network_output, activations = self.neural_net.forward(
            x=torch.tensor(sensor_data).float().reshape(1, -1),
            context=torch.tensor([[0.0]])
        )
       # Decision making influenced by genetic traits
        action_precision = self.genetic_core.physical_genetics.action_precision
        trust_baseline = self.genetic_core.heart_genetics.trust_baseline
        
        action_vector = ActionVector(selection=network_output[:, :32], parameters = network_output[:, 32:])
        # Action selection logic - now based on keys of action dictionary
        return self._select_action(action_vector, action_precision, trust_baseline, env_state)

    def _select_action(self, action_vector: ActionVector,
                      action_precision: float,
                      trust_baseline: float, env_state: EnvironmentalState) -> Tuple[str, Dict]:
       # Get action probabilities from network output
        selection = action_vector.selection
        temperature = 1.0 / action_precision
        modified_probs = np.power(selection.detach().numpy(), 1/temperature)
        modified_probs /= modified_probs.sum()


        # Select action based on modified probabilities
        action_keys = list(self.actions.keys())
        action_idx = torch.argmax(torch.tensor(modified_probs))
        selected_action_key = action_keys[action_idx]

        # Generate action parameters based on selected action
        params = self._generate_action_params(selected_action_key, trust_baseline, env_state)

        return selected_action_key, params

    def execute_action(self, action_key: str, params: Dict, env_state: EnvironmentalState) -> ActionResult:
        """Execute chosen action with genetic trait influences"""
        energy_efficiency = self.genetic_core.physical_genetics.energy_efficiency
        structural_integrity = self.genetic_core.physical_genetics.structural_integrity

        # Base energy cost modified by efficiency
        energy_cost = self._calculate_energy_cost(action_key) / energy_efficiency

        if self.energy < energy_cost:
            return ActionResult(False, -1.0, 0.0, None)

        # Action execution logic influenced by genetic traits...
        success_prob = self._calculate_success_probability(
            action_key, structural_integrity)

        # Execute action and return results...
        action_result = self._process_action_result(action_key, params, energy_cost, success_prob, env_state)
        self.energy -= energy_cost
        return action_result

    def learn_from_experience(self, env_state: EnvironmentalState, action: str, result: ActionResult):
        """Update knowledge and adapt based on action outcomes"""
        learning_efficiency = self.genetic_core.mind_genetics.learning_efficiency
        neural_plasticity = self.genetic_core.mind_genetics.neural_plasticity

        # Prepare training data
        sensor_data = self.perceive_environment(env_state)
        target_output = np.zeros(len(self.actions))
        action_index = list(self.actions.keys()).index(action)
        target_output[action_index] = result.reward
        diagnostics = self.neural_diagnostics.monitor_network_health(
            inputs=torch.tensor(sensor_data).float().detach().numpy().reshape(1, -1),
            targets=torch.tensor(target_output).float().reshape(1, -1),
            context=torch.tensor([[0.0]]),
            epoch=env_state.time_step
        )
        # Train Neural Network
        self.neural_net.backward(
            x=sensor_data.reshape(1, -1),
            y=target_output.reshape(1, -1),
            activations=None,
            learning_rate=learning_efficiency,
            plasticity=neural_plasticity
        )
        self.data_augmenter.adjust_augmentation(
            network_performance = result.reward,
            diagnostics = diagnostics
        )
        # Update performance metrics
        self._update_metrics(result)

    def get_fitness_score(self) -> float:
        """Calculate comprehensive fitness score including energy management"""
        # Base fitness from previous metrics
        base_fitness = (
            self.total_resources_gathered * 0.3 +
            self.successful_interactions * 0.2 +
            self.survival_time * 0.2 +
            self.efficiency_score * 0.2
        )

        # Energy management component
        energy_ratio = self.energy / 100.0  # Normalized to starting energy
        energy_stability = 0.1 * energy_ratio

        return base_fitness + energy_stability

    def _generate_action_params(self, action: str, trust_baseline: float, env_state: EnvironmentalState) -> Dict:
        """Generate specific parameters for each action type with genetic influence"""
        params = {}
        genetic_params = self.genetic_core.get_physical_parameters()
        brain_params = self.genetic_core.get_brain_parameters()

        if action == "move":
            # Calculate optimal direction based on resources and threats
            visible_resources = self._get_visible_resources(env_state)
            visible_threats = self._get_visible_threats(env_state)

            # Weight attractors and repulsors based on genetic traits
            direction_vector = np.zeros(2)

            for resource in visible_resources:
                weight = resource.quantity * genetic_params['sensor_resolution']
                direction = self._calculate_direction_to(resource.position, env_state)
                direction_vector += direction * weight

            for threat in visible_threats:
                weight = genetic_params['security_sensitivity']
                direction = self._calculate_direction_to(threat, env_state)
                direction_vector -= direction * weight  # Repulsion

            params['direction'] = self._normalize_vector(direction_vector)
            params['speed'] = min(2.0, self.energy / 50.0) * genetic_params['energy_efficiency']

        elif action == "gather":
            resources = self._get_visible_resources(env_state)
            if resources:
                # Score resources based on quantity, distance, and complexity
                scored_resources = []
                for resource in resources:
                    distance = self._calculate_distance(resource.position)
                    gathering_difficulty = resource.complexity / genetic_params['action_precision']
                    energy_cost = distance * gathering_difficulty

                    expected_value = (resource.quantity *
                                    genetic_params['energy_efficiency'] /
                                    energy_cost)
                    scored_resources.append((expected_value, resource))

                best_resource = max(scored_resources, key=lambda x: x[0])[1]
                params['resource_id'] = best_resource.id
                params['gather_rate'] = genetic_params['action_precision']
            else:
               params['resource_id'] = None
               params['gather_rate'] = genetic_params['action_precision']

        elif action == "process":
            params['resource_type'] = self._select_resource_to_process()
            params['processing_efficiency'] = brain_params['processing_speed']

        elif action == "share":
            params['share_amount'] = self.resources[ResourceType.ENERGY] * trust_baseline
            params['target_agent'] = self._select_sharing_target(env_state)

        elif action == "defend":
            params['defense_strength'] = self.genetic_core.heart_genetics.security_sensitivity
            params['energy_allocation'] = min(self.energy * 0.3, 30.0)
        elif action == "execute_tool":
            params['tool_name'] = 'codebase_search'
            params['tool_params'] = {"Query": "self.energy", "TargetDirectories": ['']}
            params['security_level'] = 'LOW'

        return params

    def _process_action_result(self, action: str, params: Dict, energy_cost: float, success_prob: float, env_state: EnvironmentalState) -> ActionResult:
        """Placeholder: Process gathering action and return reward"""
        success = False
        reward = 0.0
        new_state = {}

        # Assume action succeeds based on probability
        if random.random() < success_prob:
            success = True

        if action == "gather":
            reward = self._process_gathering(params, success, env_state)
        elif action == "process":
            reward = self._process_resources(params, success)
        elif action == "share":
            reward = self._process_sharing(params, success)
        elif action == "defend":
            reward = self._process_defense(params, success)
        elif action == "move":
            reward = self._process_movement(params, success)
        elif action == "execute_tool":
            reward = self._process_tool_execution(params, success)

        # Update efficiency score based on reward/energy cost ratio
        if energy_cost > 0:
           self.efficiency_score = (self.efficiency_score + max(0, reward)/energy_cost) / 2

        return ActionResult(success, reward, energy_cost, new_state)

    def _calculate_energy_cost(self, action: str) -> float:
        """Base energy cost for actions - can be adjusted based on action and genetics"""
        base_costs = {
            "move": 1.0,
            "gather": 2.0,
            "process": 5.0,
            "share": 1.5,
            "defend": 3.0,
            "execute_tool": 7.0
        }
        return base_costs.get(action, 1.0)

    def _calculate_success_probability(self, action: str, structural_integrity: float) -> float:
        """Probability of action success influenced by structural integrity"""
        base_probabilities = {
            "move": 0.95,
            "gather": 0.8,
            "process": 0.7,
            "share": 0.99,
            "defend": 0.6,
            "execute_tool": 0.9
        }
        return base_probabilities.get(action, 0.8) * structural_integrity

    def _update_metrics(self, result: ActionResult):
        """Update agent performance metrics based on action result"""
        if result.success:
            self.successful_interactions += 1
            self.total_resources_gathered += max(0, result.reward) # Only count positive resource rewards
        self.survival_time += 1

    def _calculate_distance(self, target_pos: Tuple[int, int]) -> float:
        return np.sqrt(
            (self.position[0] - target_pos[0])**2 +
            (self.position[1] - target_pos[1])**2
        )

    def _normalize_distance(self, distance, max_distance):
        return 1.0 - min(1.0, distance / max_distance) if max_distance > 0 else 0.0

    def _normalize_quantity(self, quantity):
        return min(1.0, quantity / 100.0)

    def _normalize_complexity(self, complexity):
        return min(1.0, complexity / 2.0)

    def _normalize_energy(self, energy):
        return min(1.0, energy / 100.0)

    def _get_visible_resources(self, env_state: EnvironmentalState) -> List[Resource]:
        visible_resources = []
        for resource in env_state.resources:
            distance = self._calculate_distance(resource.position)
            if distance <= 20:
                visible_resources.append(resource)
        return visible_resources

    def _get_visible_threats(self, env_state: EnvironmentalState) -> List[Tuple[int, int]]:
        visible_threats = []
        for threat_pos in env_state.threats:
            distance = self._calculate_distance(threat_pos)
            if distance <= 15:
                visible_threats.append(threat_pos)
        return visible_threats

    def _calculate_direction_to(self, target_pos: Tuple[int, int], env_state: EnvironmentalState) -> np.ndarray:
        agent_pos = np.array(self.position)
        target = np.array(target_pos)
        direction = target - agent_pos
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.array([0, 0])
        return direction / norm

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _select_resource_to_process(self) -> Optional[ResourceType]:
        if self.resources[ResourceType.MATERIALS] > 0:
            return ResourceType.MATERIALS
        elif self.resources[ResourceType.INFORMATION] > 0:
            return ResourceType.INFORMATION
        elif self.resources[ResourceType.ENERGY] > 0:
            return ResourceType.ENERGY
        return None

    def _select_sharing_target(self, env_state: EnvironmentalState) -> Optional['AdaptiveAgent']:
        nearby_agents = [agent for agent in env_state.agents if agent != self and self._calculate_distance(agent.position) < 10]
        if nearby_agents:
            return random.choice(nearby_agents)
        return None
    
    def learn_action(self, action_name: str, action_function):
      self.actions[action_name] = action_function