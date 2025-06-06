"""Statistics parsing from extracted text."""

import re
from typing import Dict, Any

from config.game_data import GameDataConfig
from .models import CharacterStats
from utils.logger import get_logger

logger = get_logger(__name__)


class StatsParser:
    """Parses character statistics from extracted text."""
    
    def __init__(self, config: GameDataConfig):
        self.config = config
    
    def parse_character_stats(self, text: str) -> CharacterStats:
        """Parse character statistics from OCR text."""
        if not text.strip():
            logger.warning("Empty text provided for parsing")
            return CharacterStats()
        
        stats = CharacterStats()
        
        # Extract all components
        stats.character_name = self._extract_character_name(text)
        stats.attribute = self._extract_attribute(text)
        stats.level = self._extract_level(text)
        stats.stats = self._extract_stats(text)
        
        logger.info(f"Parsed stats for character: {stats.character_name}")
        return stats
    
    def _extract_character_name(self, text: str) -> str:
        """Extract character name from text."""
        text_lower = text.lower()
        for name in self.config.AVAILABLE_CHARACTERS:
            if name.lower() in text_lower:
                logger.debug(f"Found character: {name}")
                return name
        return self.config.FALLBACK_CHARACTER
    
    def _extract_attribute(self, text: str) -> str:
        """Extract attribute from text."""
        text_lower = text.lower()
        for attribute in self.config.AVAILABLE_ATTRIBUTES:
            if attribute.lower() in text_lower:
                logger.debug(f"Found attribute: {attribute}")
                return attribute
        return ""
    
    def _extract_level(self, text: str) -> str:
        """Extract level from text."""
        level_match = re.search(r'Lv\.\s*(\d+)(?:/(\d+))?', text, re.IGNORECASE)
        if level_match:
            level = level_match.group(1)
            logger.debug(f"Found level: {level}")
            return level
        return ""
    
    def _extract_stats(self, text: str) -> Dict[str, Any]:
        """Extract character stats from text."""
        stats = {}
        
        for line in text.split('\n'):
            for stat_name, pattern in self.config.STAT_PATTERNS.items():
                if stat_name not in stats:  # Avoid duplicates
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        value = match.group(1)
                        stats[stat_name] = self._convert_stat_value(value)
                        logger.debug(f"Found {stat_name}: {stats[stat_name]}")
        
        return stats
    
    @staticmethod
    def _convert_stat_value(value: str) -> Any:
        """Convert string value to appropriate numeric type."""
        try:
            return float(value) if '.' in value else int(value)
        except ValueError:
            return value