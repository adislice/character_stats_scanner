"""Game-specific configuration data."""

from typing import Dict, List


class GameDataConfig:
    """Configuration for game-specific data."""
    
    AVAILABLE_CHARACTERS: List[str] = [
        "Shorekeeper", "Camellya", "Danjin", 
        "Jinhsi", "Ciaccona", "Changli", "Jiyan", 
        "Sanhua", "Yangyang", "Verina", "Xiangli Yao"
    ]
    
    AVAILABLE_ATTRIBUTES: List[str] = [
        "Spectro", "Havoc", "Electro", "Fusion", "Aero", "Glacio"
    ]
    
    STAT_PATTERNS: Dict[str, str] = {
        'HP': r'HP.*?\s+(\d+)',
        'ATK': r'ATK.*?\s+(\d+)',
        'DEF': r'DEF.*?\s+(\d+)',
        'Energy Regen': r'Energy.*?Regen.*?([\d.]+)',
        'Crit. Rate': r'Crit.*?Rate.*?([\d.]+)',
        'Crit. DMG': r'Crit.*?DMG.*?([\d.]+)'
    }
    
    FALLBACK_CHARACTER: str = "Rover"
    
    # OCR Configuration
    OCR_CONFIG: str = '--psm 6'
    
    # Template matching parameters
    MIN_MATCHES: int = 4
    RANSAC_THRESHOLD: float = 5.0