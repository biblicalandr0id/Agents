import random

class EmbryoNamer:
    def __init__(self):
        self.prefixes = ["Neo", "Proto", "Meta", "Hyper", "Sub", "Quantum", "Astro", "Cosmo", "Bio", "Nano", "Syntho"]
        self.suffixes = [
            "Wave", "Flux", "Chain", "Core", "Point", "Drive", "Pulse", "Node", "Nexus",
            "Matrix", "Sphere", "Grid", "Vector", "Field", "Array", "Link", "Cloud", "Net",
            "Flow", "Stream", "Circuit", "Engine", "Source", "System", "Unit", "Hub", "Module"
        ]

    def generate_random_name(self) -> str:
      prefix = random.choice(self.prefixes)
      suffix = random.choice(self.suffixes)
      number = str(random.randint(100, 999))
      letter = chr(random.randint(65, 90)) # Generate a random uppercase letter (A-Z)
      return f"{prefix}-{suffix}-{letter}{number}"