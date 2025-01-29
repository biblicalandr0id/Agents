import os
from datetime import datetime
from embryo_namer import EmbryoNamer
from genetics import GeneticCore
import json

def generate_embryo(genetic_data, output_dir="embryos"):
    """Generates the embryo based on the agent's genetic data, adding import and initialization logic"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    embryo_name = f"embryo_{EmbryoNamer().generate_random_name()}_{timestamp}"
    file_path = os.path.join(output_dir, f"{embryo_name}.py")

    os.makedirs(output_dir, exist_ok=True)

    with open(file_path, 'w') as f:
        f.write("from typing import List, Tuple, Dict, Optional\n")
        f.write("import numpy as np\n")
        f.write("import torch\n")
        f.write("import random\n")
        f.write("import math\n")
        f.write("from agent_environment import ActionVector, ActionResult, EnvironmentalState\n")
        f.write("from genetics import GeneticCore\n")
        f.write("from neural_networks import NeuralAdaptiveNetwork\n")
        f.write("from executor import AdaptiveExecutor\n")
        f.write("from diagnostics import NeuralDiagnostics\n")
        f.write("from augmentation import AdaptiveDataAugmenter\n")

        f.write("\n")
        f.write("class AutonomousEmbryo:\n")
        f.write("    def __init__(self, genetic_data,  position: Tuple[int, int] = (0, 0)):\n")
        f.write("        self.genetic_core = GeneticCore()\n")
        f.write("        self.genetic_core.load_genetics(genetic_data)\n")
        f.write("        self.position = position\n")
        f.write("        self.energy = 100.0\n")
        f.write("        self.resources = {}\n")
        f.write("        self.knowledge_base = {}\n")
        f.write("        self.data_augmenter = AdaptiveDataAugmenter()\n")
        f.write("        self.neural_diagnostics = NeuralDiagnostics()\n")
        f.write("    def get_status(self):\n")
        f.write("        return {\n")
        f.write("            'embryo_id':  \"{}\",\n".format(embryo_name.split('_')[1]))
        f.write("            'age': 0,\n")
        f.write("            'development_stage': self.genetic_core.development_progress,\n")
        f.write("            'experiences_count': 0,\n")
        f.write("            'neural_connections': {'cognitive': 100, 'behavioral': 100, 'processing': 100},\n")
        f.write("            'specializations': {},\n")
        f.write("            'potential_capabilities': {\n")
        f.write("                'adaptability_index': self.genetic_core.base_traits.adaptability,\n")
        f.write("                'resilience_factor': self.genetic_core.base_traits.resilience,\n")
        f.write("                'processing_power': self.genetic_core.brain_genetics.processing_speed,\n")
        f.write("                'efficiency_level': self.genetic_core.physical_genetics.energy_efficiency,\n")
        f.write("            }\n")
        f.write("        }\n")
        f.write("    def learn_from_experience(self, env_state: EnvironmentalState, action: str, result: ActionResult):\n")
        f.write("       pass\n")
        f.write("    def decide_action(self, env_state: EnvironmentalState) -> Tuple[str, Dict]:\n")
        f.write("       pass\n")
        f.write("    def execute_action(self, action: str, params: Dict, env_state: EnvironmentalState) -> ActionResult:\n")
        f.write("       pass\n")


    return file_path

def generate_embryo_file(genetic_core, output_dir="embryos"):
    """Helper function to generate the embryo with GeneticCore object"""
    return generate_embryo(genetic_core.save_genetics(), output_dir)