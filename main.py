"""Main entry point for the character stats extractor."""

import sys
from pathlib import Path

import cv2

from extractor.character_extractor import CharacterStatsExtractor
from utils.logger import setup_logger


# Setup logging
logger = setup_logger(__name__)


def main():
    """Main function for standalone execution."""
    # Configuration
    screenshot_path = "screenshots/Screenshot 2025-06-06 165153.png"
    template_path = "templates/stats_template.png"
    
    try:
        # Validate input files
        if not Path(screenshot_path).exists():
            logger.error(f"Screenshot file not found: {screenshot_path}")
            return 1
        
        if not Path(template_path).exists():
            logger.error(f"Template file not found: {template_path}")
            return 1
        
        # Initialize extractor
        extractor = CharacterStatsExtractor()
        
        # Extract stats
        result_json = extractor.extract_to_json(
            screenshot_path, 
            template_path, 
            debug=True
        )
        
        print("Extracted Character Stats:")
        print(result_json)
        
        logger.info("Extraction completed successfully")
        cv2.waitKey(0)
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())