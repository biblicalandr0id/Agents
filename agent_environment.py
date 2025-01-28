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
from adaptive_environment import AdaptiveEnvironment, EnvironmentalState, Resource, ResourceType
from perlin_noise import PerlinNoise

class ResourceType(Enum):
    ENERGY = "energy"
    INFORMATION = "information"
    MATERIALS = "materials"

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

class AgentAction(Enum):
    MOVE = "move"
    GATHER = "gather"
    PROCESS = "process"
    SHARE = "share"
    DEFEND = "defend"
    EXECUTE_TOOL = "execute_tool"

@dataclass
class ActionResult:
    success: bool
    reward: float
    energy_cost: float
    new_state: Optional[Dict]

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

        return augmented_inputs

    def decide_action(self, env_state: EnvironmentalState) -> Tuple[AgentAction, Dict]:
        sensor_data = self.perceive_environment(env_state)

        genetic_modifiers = {
            'processing_speed': self.genetic_core.brain_genetics.processing_speed,
            'sensor_sensitivity': self.genetic_core.physical_genetics.sensor_sensitivity
        }

        network_output, activations = self.neural_net.forward(
            x=sensor_data.reshape(1, -1),
            context=torch.tensor([[0.0]])
        )
        network_output = network_output.flatten()

        action_precision = self.genetic_core.physical_genetics.action_precision
        trust_baseline = self.genetic_core.heart_genetics.trust_baseline

        return self._select_action(network_output, action_precision, trust_baseline, env_state) # Pass env_state

    def execute_action(self, action: AgentAction, params: Dict) -> ActionResult:
        energy_efficiency = self.genetic_core.physical_genetics.energy_efficiency
        structural_integrity = self.genetic_core.physical_genetics.structural_integrity

        energy_cost = self._calculate_energy_cost(action) / energy_efficiency

        if self.energy < energy_cost:
            return ActionResult(False, -1.0, 0.0, None)

        success_prob = self._calculate_success_probability(
            action, structural_integrity)

        action_result = self._process_action_result(action, params, energy_cost, success_prob, env_state) # Pass env_state
        self.energy -= energy_cost
        return action_result

    def learn_from_experience(self, env_state: EnvironmentalState, action: AgentAction, result: ActionResult):
        learning_efficiency = self.genetic_core.mind_genetics.learning_efficiency
        neural_plasticity = self.genetic_core.mind_genetics.neural_plasticity

        sensor_data = self.perceive_environment(env_state)
        target_output = np.zeros(len(AgentAction))
        action_index = list(AgentAction).index(action)
        target_output[action_index] = result.reward
        diagnostics = self.neural_diagnostics.monitor_network_health(
            inputs=torch.tensor(sensor_data).float().reshape(1, -1),
            targets=torch.tensor(target_output).float().reshape(1, -1),
            context=torch.tensor([[0.0]]),
            epoch=env_state.time_step
        )
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
        self._update_metrics(result)

    def get_fitness_score(self) -> float:
        base_fitness = (
            self.total_resources_gathered * 0.3 +
            self.successful_interactions * 0.2 +
            self.survival_time * 0.2 +
            self.efficiency_score * 0.2
        )

        energy_ratio = self.energy / 100.0
        energy_stability = 0.1 * energy_ratio

        return base_fitness + energy_stability

    def _select_action(self, network_output: np.ndarray,
        action_precision: float,
        trust_baseline: float, env_state: EnvironmentalState) -> Tuple[AgentAction, Dict]: # Pass env_state
        action_probs = network_output

        temperature = 1.0 / action_precision
        modified_probs = np.power(action_probs, 1/temperature)
        modified_probs /= modified_probs.sum()

        action_idx = np.random.choice(len(AgentAction), p=modified_probs)
        selected_action = list(AgentAction)[action_idx]

        params = self._generate_action_params(selected_action, trust_baseline, env_state) # Pass env_state

        return selected_action, params

    def _generate_action_params(self, action: AgentAction, trust_baseline: float, env_state: EnvironmentalState) -> Dict: # Pass env_state
        params = {}
        genetic_params = self.genetic_core.get_physical_parameters()
        brain_params = self.genetic_core.get_brain_parameters()

        if action == AgentAction.MOVE:
            visible_resources = self._get_visible_resources(env_state)
            visible_threats = self._get_visible_threats(env_state)

            direction_vector = np.zeros(2)

            for resource in visible_resources:
                weight = resource.quantity * genetic_params['sensor_resolution']
                direction = self._calculate_direction_to(resource.position, env_state)
                direction_vector += direction * weight

            for threat in visible_threats:
                weight = genetic_params['security_sensitivity']
                direction = self._calculate_direction_to(threat, env_state)
                direction_vector -= direction * weight

            params['direction'] = self._normalize_vector(direction_vector)
            params['speed'] = min(2.0, self.energy / 50.0) * genetic_params['energy_efficiency']

        elif action == AgentAction.GATHER:
            resources = self._get_visible_resources(env_state)
            if resources:
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

        elif action == AgentAction.PROCESS:
            params['resource_type'] = self._select_resource_to_process()
            params['processing_efficiency'] = brain_params['processing_speed']

        elif action == AgentAction.SHARE:
            params['share_amount'] = self.resources[ResourceType.ENERGY] * trust_baseline
            params['target_agent'] = self._select_sharing_target(env_state)

        elif action == AgentAction.DEFEND:
            params['defense_strength'] = self.genetic_core.heart_genetics.security_sensitivity
            params['energy_allocation'] = min(self.energy * 0.3, 30.0)
        elif action == AgentAction.EXECUTE_TOOL:
            params['tool_name'] = 'codebase_search'
            params['tool_params'] = {"Query": "self.energy", "TargetDirectories": ['']}
            params['security_level'] = 'LOW'

        return params

    def _process_action_result(self, action: AgentAction, params: Dict, energy_cost: float, success_prob: float, env_state: EnvironmentalState) -> ActionResult: # Pass env_state
        success = False
        reward = 0.0
        new_state = {}

        if random.random() < success_prob:
            success = True

        if action == AgentAction.GATHER:
            reward = self._process_gathering(params, success, env_state)
        elif action == AgentAction.PROCESS:
            reward = self._process_resources(params, success)
        elif action == AgentAction.SHARE:
            reward = self._process_sharing(params, success)
        elif action == AgentAction.DEFEND:
            reward = self._process_defense(params, success)
        elif action == AgentAction.MOVE:
            reward = self._process_movement(params, success)
        elif action == AgentAction.EXECUTE_TOOL:
            reward = self._process_tool_execution(params, success)

        if energy_cost > 0:
            self.efficiency_score = (self.efficiency_score + max(0, reward)/energy_cost) / 2

        return ActionResult(success, reward, energy_cost, new_state)

    def _calculate_energy_cost(self, action: AgentAction) -> float:
        base_costs = {
            AgentAction.MOVE: 1.0,
            AgentAction.GATHER: 2.0,
            AgentAction.PROCESS: 5.0,
            AgentAction.SHARE: 1.5,
            AgentAction.DEFEND: 3.0,
            AgentAction.EXECUTE_TOOL: 7.0
        }
        return base_costs.get(action, 1.0)

    def _calculate_success_probability(self, action: AgentAction, structural_integrity: float) -> float:
        base_probabilities = {
            AgentAction.MOVE: 0.95,
            AgentAction.GATHER: 0.8,
            AgentAction.PROCESS: 0.7,
            AgentAction.SHARE: 0.99,
            AgentAction.DEFEND: 0.6,
            AgentAction.EXECUTE_TOOL: 0.9
        }
        return base_probabilities.get(action, 0.8) * structural_integrity

    def _update_metrics(self, result: ActionResult):
        if result.success:
            self.successful_interactions += 1
            self.total_resources_gathered += max(0, result.reward)
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

    def _process_gathering(self, params: Dict, success: bool, env_state: EnvironmentalState) -> float:
        if success and params['resource_id']:
            resource_id = params['resource_id']
            for resource in env_state.resources:
                if resource.id == resource_id:
                    gathered_quantity = min(resource.quantity, params['gather_rate'])
                    self.resources[resource.type] += gathered_quantity
                    resource.quantity -= gathered_quantity
                    return gathered_quantity
        return -0.1

    def _process_resources(self, params: Dict, success: bool) -> float:
        if success and params['resource_type']:
            resource_type = params['resource_type']
            if self.resources[resource_type] > 0:
                processing_rate = params['processing_efficiency']
                processed_quantity = self.resources[resource_type] * processing_rate
                self.resources[resource_type] -= processed_quantity
                self.energy += processed_quantity * 10
                return processed_quantity * 5
        return -0.5

    def _process_sharing(self, params: Dict, success: bool) -> float:
        if success and params['target_agent']:
            share_amount = params['share_amount']
            target_agent = params['target_agent']
            if self.resources[ResourceType.ENERGY] >= share_amount:
                self.resources[ResourceType.ENERGY] -= share_amount
                target_agent.energy += share_amount
                return share_amount
        return -0.2

    def _process_defense(self, params: Dict, success: bool) -> float:
        if success:
            defense_strength = params['defense_strength']
            energy_invested = params['energy_allocation']
            self.energy -= energy_invested
            return defense_strength
        return -0.3

    def _process_movement(self, params: Dict, success: bool) -> float:
        if success:
            direction = params['direction']
            speed = params['speed']
            new_position = (self.position[0] + direction[0] * speed, self.position[1] + direction[1] * speed)
            self.position = new_position
            return 0.01
        return -0.05

    def _process_tool_execution(self, params: Dict, success: bool) -> float:
        if success and params['tool_name']:
            tool_name = params['tool_name']
            if tool_name == 'codebase_search':
                return 1.0
        return -0.8

class EnhancedAdaptiveEnvironment(AdaptiveEnvironment):
    def __init__(self, size: Tuple[int, int], complexity: float):
        super().__init__(size, complexity) # ADDED super().__init__(size, complexity)
        self.terrain = self._generate_terrain()
        self.weather = self._initialize_weather()
        self.agents = []

    def _generate_terrain(self) -> np.ndarray:
        size_x, size_y = self.size
        terrain = np.zeros(self.size)
        scale = 10
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0

        for i in range(octaves):
            frequency = lacunarity ** i
            amplitude = persistence ** i
            x_coords = np.linspace(0, size_x / scale * frequency, size_x)
            y_coords = np.linspace(0, size_y / scale * frequency, size_y)
            xv, yv = np.meshgrid(x_coords, y_coords)
            noise = PerlinNoise(octaves=octaves) 
            sample = np.array([[noise([x, y]) for x in x_coords] for y in y_coords])
            terrain += amplitude * sample

        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
        return terrain

    def _update_state(self):
        self.current_state.time_step += 1

        for resource in self.current_state.resources:
            if resource.quantity < 100:
                resource.quantity += random.uniform(0, 0.5)
            resource.position = (
                max(0, min(self.size[0] - 1, int(resource.position[0] + random.uniform(-1, 1)))),
                max(0, min(self.size[1] - 1, int(resource.position[1] + random.uniform(-1, 1))))
            )

        for i in range(len(self.current_state.threats)):
            threat_pos = self.current_state.threats[i]
            nearest_agent = self._find_nearest_agent(threat_pos)
            if nearest_agent:
                direction = self._calculate_direction_to(threat_pos, nearest_agent.position)
                new_threat_pos = (threat_pos[0] + direction[0], threat_pos[1] + direction[1])
                self.current_state.threats[i] = (max(0, min(self.size[0]-1, int(new_threat_pos[0]))), max(0, min(self.size[1]-1, int(new_threat_pos[1]))))
            else:
                self.current_state.threats[i] = (max(0, min(self.size[0]-1, int(threat_pos[0] + random.uniform(-1, 1)))), max(0, min(self.size[1]-1, int(threat_pos[1] + random.uniform(-1, 1)))))


        if random.random() < 0.01 * self.current_state.complexity_level:
            self.current_state.resources.append(
                Resource(
                    type=ResourceType.ENERGY,
                    quantity=random.uniform(10, 50),
                    position=(random.randint(0, self.size[0]-1), random.randint(0, self.size[1]-1)),
                    complexity=random.uniform(0.1, 0.9)
                )
            )
        self.current_state.agents = self.agents


    def _calculate_threat_movement(self, threat_pos: Tuple[float, float]) -> Tuple[float, float]:
        return (random.uniform(-1, 1), random.uniform(-1, 1))

    def _find_nearest_agent(self, pos: Tuple[float, float]) -> Optional['AdaptiveAgent']:
        min_distance = float('inf')
        nearest_agent = None
        for agent in self.agents:
            distance = self._calculate_distance(pos, agent.position)
            if distance < min_distance:
                min_distance = distance
                nearest_agent = agent
        return nearest_agent

    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


    def _generate_perlin_noise(self, size: Tuple[int, int], scale: float) -> np.ndarray:
        return np.zeros(size)

    def _initialize_weather(self) -> Dict:
        return {}

    def _update_weather(self, current_weather: Dict) -> Dict:
        return current_weather

    def _get_terrain_factor(self, position: Tuple[int, int]) -> float:
        return 1.0

    def _get_weather_factor(self, position: Tuple[int, int]) -> float:
        return 1.0

    def _calculate_terrain_gradient(self, position: Tuple[int, int]) -> Tuple[float, float]:
        return (0.0, 0.0)

    def _find_nearest_agent(self, pos: Tuple[float, float]) -> Optional['AdaptiveAgent']:
        return None