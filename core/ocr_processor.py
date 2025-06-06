"""OCR processing functionality."""

import os
import pytesseract
import numpy as np
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class OCRProcessor:
    """Handles OCR operations and text extraction."""
    
    def __init__(self, tesseract_path: Optional[str] = None, config: str = '--psm 6'):
        """Initialize OCR processor with optional Tesseract path."""
        self.config = config
        self._setup_tesseract_path(tesseract_path)
    
    def _setup_tesseract_path(self, tesseract_path: Optional[str]) -> None:
        """Setup Tesseract executable path."""
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        elif os.name == 'nt':  # Windows
            default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(default_path):
                pytesseract.pytesseract.tesseract_cmd = default_path
                logger.info(f"Using default Tesseract path: {default_path}")
    
    def extract_text(self, image: np.ndarray) -> str:
        """Extract text from image using OCR."""
        try:
            text = pytesseract.image_to_string(image, config=self.config)
            logger.debug(f"Extracted text: {text}")
            return text.strip()
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""