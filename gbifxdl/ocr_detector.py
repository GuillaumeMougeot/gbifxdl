# OCR Text Detection Module for GBIFXDL
# This module provides text detection capabilities to identify and optionally
# filter out images containing text (labels, barcodes, etc.) from biological
# specimen images.

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None

__all__ = ["OCRDetector", "detect_text_in_image"]


class OCRDetector:
    """
    OCR-based text detector for filtering images with text content.
    
    Uses pytesseract to detect text in images. Primarily useful for filtering
    out specimen images with labels, barcodes, or significant text overlays.
    
    Parameters
    ----------
    confidence_threshold : float, default=60.0
        Minimum confidence score (0-100) for text detection. Lower values
        are more sensitive but may produce false positives.
    min_text_length : int, default=3
        Minimum number of characters to consider as valid text. Helps
        filter out noise.
    lang : str, default='eng'
        Tesseract language code. Use 'eng' for English, or combine multiple
        languages like 'eng+fra'.
    psm : int, default=3
        Page Segmentation Mode for tesseract. 3 = Fully automatic page
        segmentation (default). See tesseract docs for other options.
    logger : logging.Logger, optional
        Logger instance for debugging.
        
    Attributes
    ----------
    available : bool
        Whether pytesseract is available for use.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 60.0,
        min_text_length: int = 3,
        lang: str = 'eng',
        psm: int = 3,
        logger: Optional[logging.Logger] = None,
    ):
        if pytesseract is None:
            raise ImportError(
                "pytesseract is not installed. "
                "Install it with: pip install pytesseract\n"
                "Also ensure tesseract-ocr is installed on your system:\n"
                "  Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
                "  macOS: brew install tesseract\n"
                "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
            )
        
        self.confidence_threshold = confidence_threshold
        self.min_text_length = min_text_length
        self.lang = lang
        self.psm = psm
        self.logger = logger or logging.getLogger(__name__)
        self.available = True
        
        # Test tesseract availability
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            self.logger.warning(f"Tesseract not properly installed: {e}")
            self.available = False
    
    def detect_text(
        self,
        image: Image.Image,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Detect text in an image.
        
        Parameters
        ----------
        image : PIL.Image.Image
            Image to analyze for text.
        return_details : bool, default=False
            If True, return detailed OCR results including detected text
            and confidence scores.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'has_text' (bool): Whether significant text was detected
            - 'text_confidence' (float): Average confidence of detected text
            - 'num_words' (int): Number of words detected
            - 'detected_text' (str, optional): Extracted text if return_details=True
            - 'word_confidences' (list, optional): Per-word confidence scores
        """
        if not self.available:
            self.logger.warning("Tesseract not available, skipping OCR")
            return {
                'has_text': False,
                'text_confidence': 0.0,
                'num_words': 0,
            }
        
        try:
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                config=f'--psm {self.psm}',
                output_type=pytesseract.Output.DICT
            )
            
            # Filter by confidence threshold
            valid_words = []
            valid_confidences = []
            
            for i, conf in enumerate(ocr_data['conf']):
                if conf != -1 and int(conf) >= self.confidence_threshold:
                    text = ocr_data['text'][i].strip()
                    if len(text) >= self.min_text_length:
                        valid_words.append(text)
                        valid_confidences.append(int(conf))
            
            has_text = len(valid_words) > 0
            avg_confidence = np.mean(valid_confidences) if valid_confidences else 0.0
            
            result = {
                'has_text': has_text,
                'text_confidence': float(avg_confidence),
                'num_words': len(valid_words),
            }
            
            if return_details:
                result['detected_text'] = ' '.join(valid_words)
                result['word_confidences'] = valid_confidences
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during OCR processing: {e}")
            return {
                'has_text': False,
                'text_confidence': 0.0,
                'num_words': 0,
                'error': str(e),
            }
    
    def detect_text_from_path(
        self,
        image_path: str,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Detect text in an image file.
        
        Parameters
        ----------
        image_path : str
            Path to the image file.
        return_details : bool, default=False
            If True, return detailed OCR results.
            
        Returns
        -------
        dict
            OCR detection results (see detect_text method).
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                return self.detect_text(img, return_details=return_details)
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return {
                'has_text': False,
                'text_confidence': 0.0,
                'num_words': 0,
                'error': str(e),
            }


def detect_text_in_image(
    image_path: str,
    confidence_threshold: float = 60.0,
    min_text_length: int = 3,
) -> bool:
    """
    Simple function to check if an image contains text.
    
    Parameters
    ----------
    image_path : str
        Path to the image file.
    confidence_threshold : float, default=60.0
        Minimum confidence score for text detection.
    min_text_length : int, default=3
        Minimum number of characters to consider as valid text.
        
    Returns
    -------
    bool
        True if text was detected, False otherwise.
    """
    detector = OCRDetector(
        confidence_threshold=confidence_threshold,
        min_text_length=min_text_length
    )
    result = detector.detect_text_from_path(image_path)
    return result.get('has_text', False)
