"""
SM-MCPM Configuration Management
Load and validate YAML configuration files
"""

import yaml
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate configuration
    validate_config(config)
    
    print(f" Loaded config from {config_path}")
    return config


def validate_config(config):
    """
    Validate configuration parameters
    
    Raises:
        AssertionError: If config is invalid
    """
    # Simulation
    assert config['simulation']['width'] > 0, "Width must be positive"
    assert config['simulation']['height'] > 0, "Height must be positive"
    assert 0 < config['simulation']['dt'] < 1, "dt must be in (0, 1)"
    
    # Agents
    assert config['agents']['count'] > 0, "Agent count must be positive"
    assert config['agents']['base_speed'] > 0, "Speed must be positive"
    
    # Energy
    assert config['energy']['hunger_decay'] > 0, "Hunger decay must be positive"
    assert config['energy']['scouting_multiplier'] >= 1.0, "Scouting multiplier must be >= 1"
    
    # Placement weights
    placement = config['placement']
    alpha = placement['death_weight']
    beta = placement['trail_weight']
    assert 0 <= alpha <= 1, "death_weight must be in [0, 1]"
    assert 0 <= beta <= 1, "trail_weight must be in [0, 1]"
    assert abs(alpha + beta - 1.0) < 0.01, "death_weight + trail_weight should sum to 1"
    
    print(" Config validation passed")


if __name__ == "__main__":
    # Test loading default config
    try:
        config = load_config('configs/default.yaml')
        print(" config.py ready")
    except FileNotFoundError:
        print(" Default config not found")