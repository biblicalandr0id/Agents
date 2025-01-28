from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import uuid
import numpy as np
import math
import random

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

class AdaptiveEnvironment:
    def __init__(self, size: Tuple[int, int], complexity: float):
        self.size = size
        self.complexity = complexity
        self.current_state = EnvironmentalState(
            resources=[],
            threats=[],
            time_step=0,
            complexity_level=complexity
        )
        self.agents = [] # Initialize agents list

    def step(self, agents):
        """Advance environment by one time step, updating resources, threats, and agent interactions."""
        self._update_state()
        agent_actions_results = []
        for agent in agents:
            action, params = agent.decide_action(self.current_state)
            result = agent.execute_action(action, params)
            agent.learn_from_experience(self.current_state, action, result)
            agent_actions_results.append((agent, action, result))
        return self.current_state, agent_actions_results

    def _update_state(self):
        """Basic state update - can be expanded in subclasses."""
        self.current_state.time_step += 1
        # Add basic resource regeneration (example)
        for resource in self.current_state.resources:
            if resource.quantity < 100:
                resource.quantity += random.uniform(0, 0.1) # Slow regeneration

    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)