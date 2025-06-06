"""Data models for character statistics."""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class CharacterStats:
    """Data class for character statistics."""
    character_name: str = ""
    attribute: str = ""
    level: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'character_name': self.character_name,
            'attribute': self.attribute,
            'level': self.level,
            'stats': self.stats
        }

    def is_valid(self) -> bool:
        """Check if the extracted stats contain meaningful data."""
        return bool(self.character_name or self.stats)