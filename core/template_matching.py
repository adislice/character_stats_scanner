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
        # x, y, w, h = self._match_features_orb(img, template)
        x, y, w, h = self._match_features_sift(img, template)
        
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
        
    def _match_features_sift(self, img: np.ndarray, template: np.ndarray) -> Tuple[int, int, int, int]:
        """Perform SIFT feature matching and return bounding box."""
        # SIFT feature detection
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(template, None)
        kp2, des2 = sift.detectAndCompute(img, None)

        if des1 is None or des2 is None:
            raise ValueError("No features detected in images")

        # Feature matching dengan FLANN (lebih cocok untuk float descriptor seperti SIFT)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < self.min_matches:
            raise ValueError(f"Not enough matches found: {len(good_matches)} < {self.min_matches}")

        # Homography calculation
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
        if M is None:
            raise ValueError("Failed to compute homography")

        # Transform template corners
        h, w = template.shape
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(dst)

        match_img = cv2.drawMatches(template, kp1, img, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Template Match", match_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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