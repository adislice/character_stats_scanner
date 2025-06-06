"""Main character statistics extractor."""

import json
import cv2
from typing import Optional

from core.models import CharacterStats
from core.ocr_processor import OCRProcessor
from core.template_matching import TemplateMatching
from core.stats_parser import StatsParser
from config.game_data import GameDataConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class CharacterStatsExtractor:
    """Main class for extracting character statistics from screenshots."""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """Initialize the extractor with dependencies."""
        self.config = GameDataConfig()
        self.ocr = OCRProcessor(tesseract_path, self.config.OCR_CONFIG)
        self.template_matcher = TemplateMatching(
            self.config.MIN_MATCHES, 
            self.config.RANSAC_THRESHOLD
        )
        self.parser = StatsParser(self.config)
    
    def extract_from_screenshot(self, screenshot_path: str, template_path: str, 
                              debug: bool = False) -> CharacterStats:
        """Extract character statistics from a screenshot."""
        try:
            logger.info(f"Extracting stats from: {screenshot_path}")
            
            # Crop the character stats region
            cropped_img = self.template_matcher.crop_detected_region(
                screenshot_path, template_path, debug
            )
            
            # Extract text using OCR
            text = self.ocr.extract_text(cropped_img)
            
            # Parse stats from text
            stats = self.parser.parse_character_stats(text)
            
            # Show cropped image if debug mode
            if debug:
                cv2.imshow("Cropped Result", cropped_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            if not stats.is_valid():
                logger.warning("No meaningful stats extracted")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to extract stats: {e}")
            raise
    
    def extract_to_json(self, screenshot_path: str, template_path: str, 
                       output_path: Optional[str] = None, debug: bool = False) -> str:
        """Extract stats and return as JSON string."""
        stats = self.extract_from_screenshot(screenshot_path, template_path, debug)
        json_str = json.dumps(stats.to_dict(), indent=2)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(json_str)
            logger.info(f"Results saved to {output_path}")
        
        return json_str