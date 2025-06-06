"""Template matching functionality for icon detection."""

import cv2
import numpy as np
import os
from typing import Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


class TemplateMatching:
    """Handles template matching operations for icon detection."""
    
    def __init__(self, min_matches: int = 4, ransac_threshold: float = 5.0):
        """Initialize template matching with parameters."""
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
    
    def detect_icon_position(self, screenshot_path: str, template_path: str, 
                           debug: bool = False) -> Tuple[int, int, int, int]:
        """Detect icon position using ORB feature matching."""
        self._validate_files(screenshot_path, template_path)
        
        # Load images
        img = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or template is None:
            raise ValueError("Failed to load images")
        
        # Perform feature matching
        x, y, w, h = self._match_features_orb(img, template)
        
        # Debug visualization
        if debug:
            self._show_debug_visualization(screenshot_path, x, y, w, h)
        
        return x, y, w, h
    
    def crop_detected_region(self, screenshot_path: str, template_path: str, 
                           debug: bool = False) -> np.ndarray:
        """Crop the detected region from screenshot."""
        x, y, w, h = self.detect_icon_position(screenshot_path, template_path, debug)
        
        original_img = cv2.imread(screenshot_path)
        if original_img is None:
            raise ValueError(f"Failed to load screenshot: {screenshot_path}")
        
        cropped_img = original_img[y:y+h, x:x+w]
        logger.info(f"Cropped region: ({x}, {y}, {w}, {h})")
        return cropped_img
    
    @staticmethod
    def _validate_files(screenshot_path: str, template_path: str) -> None:
        """Validate that required files exist."""
        if not os.path.exists(screenshot_path):
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")
    
    def _match_features_orb(self, img: np.ndarray, template: np.ndarray) -> Tuple[int, int, int, int]:
        """Perform ORB feature matching and return bounding box."""
        # ORB feature detection
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(img, None)
        
        if des1 is None or des2 is None:
            raise ValueError("No features detected in images")
        
        # Feature matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < self.min_matches:
            raise ValueError(f"Not enough matches found: {len(matches)} < {self.min_matches}")
        
        # Homography calculation
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
        if M is None:
            raise ValueError("Failed to compute homography")
        
        # Transform template corners
        h, w = template.shape
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(dst)

        return x, y, w, h
    
    @staticmethod
    def _show_debug_visualization(screenshot_path: str, x: int, y: int, w: int, h: int) -> None:
        """Show debug visualization of detected region."""
        debug_img = cv2.imread(screenshot_path)
        if debug_img is not None:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.imshow("Template Match Debug", debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()