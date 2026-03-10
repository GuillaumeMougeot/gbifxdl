# YOLO Object Detection Module for GBIFXDL
# This module provides object detection and cropping capabilities for biological
# specimen images using YOLO models from Ultralytics.


import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from PIL import Image
import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

__all__ = ["YOLODetector", "detect_and_crop"]


class YOLODetector:
    """
    YOLO-based object detector and cropper for specimen images.
    
    Uses Ultralytics YOLO models to detect objects (e.g., arthropods) in
    images and crop to the best bounding box. Supports loading models from
    local paths or Hugging Face Hub.
    
    Parameters
    ----------
    model_path : str, optional
        Path to local YOLO model weights (.pt file). If None, must provide
        model_repo and model_filename for HuggingFace download.
    model_repo : str, optional
        Hugging Face repository ID (e.g., "edgaremy/arthropod-detector").
        Used only if model_path is None.
    model_filename : str, optional
        Filename of model weights in the HuggingFace repo.
    device : str, default='cpu'
        Device for inference ('cpu', 'cuda', 'cuda:0', etc.).
    conf_threshold : float, default=0.25
        Confidence threshold for detections (0-1).
    iou_threshold : float, default=0.45
        IoU threshold for NMS (non-maximum suppression).
    imgsz : int, default=640
        Image size for inference. Higher values may improve accuracy but
        are slower.
    logger : logging.Logger, optional
        Logger instance for debugging.
        
    Attributes
    ----------
    available : bool
        Whether YOLO is available for use.
    model : YOLO
        Loaded YOLO model instance.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_repo: Optional[str] = None,
        model_filename: Optional[str] = None,
        device: str = 'cpu',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        logger: Optional[logging.Logger] = None,
    ):
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed. "
                "Install it with: pip install ultralytics"
            )
        
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.logger = logger or logging.getLogger(__name__)
        self.available = True
        self.model = None
        
        # Load model
        try:
            if model_path is not None:
                # Load from local path
                self.logger.info(f"Loading YOLO model from {model_path}")
                self.model = YOLO(model_path)
            elif model_repo is not None and model_filename is not None:
                # Download from Hugging Face
                try:
                    from huggingface_hub import hf_hub_download
                except ImportError:
                    raise ImportError(
                        "huggingface_hub is not installed. "
                        "Install it with: pip install huggingface_hub"
                    )
                
                self.logger.info(f"Downloading model from {model_repo}/{model_filename}")
                weights_path = hf_hub_download(
                    repo_id=model_repo,
                    filename=model_filename
                )
                self.model = YOLO(weights_path)
            else:
                raise ValueError(
                    "Must provide either model_path or both model_repo and model_filename"
                )
            
            # Move to device
            self.model.to(device)
            self.logger.info(f"YOLO model loaded successfully on {device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.available = False
    
    def detect(
        self,
        image: Union[Image.Image, str, np.ndarray],
        return_all: bool = False
    ) -> Dict[str, Any]:
        """
        Detect objects in an image.
        
        Parameters
        ----------
        image : PIL.Image.Image, str, or np.ndarray
            Image to analyze. Can be a PIL Image, path to image file,
            or numpy array.
        return_all : bool, default=False
            If True, return all detections. If False, return only the
            best detection (highest confidence * area).
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'detected' (bool): Whether any objects were detected
            - 'num_detections' (int): Number of objects detected
            - 'best_box' (list): [x1, y1, x2, y2] of best detection
            - 'best_conf' (float): Confidence of best detection
            - 'best_class' (int): Class ID of best detection
            - 'all_boxes' (list, optional): All bounding boxes if return_all=True
            - 'all_confs' (list, optional): All confidences if return_all=True
        """
        if not self.available or self.model is None:
            self.logger.warning("YOLO model not available, skipping detection")
            return {
                'detected': False,
                'num_detections': 0,
                'best_box': None,
                'best_conf': 0.0,
            }
        
        try:
            # Run inference
            results = self.model.predict(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device,
            )
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                return {
                    'detected': False,
                    'num_detections': 0,
                    'best_box': None,
                    'best_conf': 0.0,
                }
            
            # Extract boxes and confidences
            boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            # Calculate score as confidence * area for each box
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            scores = confs * areas
            
            # Get best detection
            best_idx = np.argmax(scores)
            best_box = boxes[best_idx].tolist()
            best_conf = float(confs[best_idx])
            best_class = int(classes[best_idx])
            
            result = {
                'detected': True,
                'num_detections': len(boxes),
                'best_box': best_box,
                'best_conf': best_conf,
                'best_class': best_class,
            }
            
            if return_all:
                result['all_boxes'] = boxes.tolist()
                result['all_confs'] = confs.tolist()
                result['all_classes'] = classes.tolist()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during YOLO detection: {e}")
            return {
                'detected': False,
                'num_detections': 0,
                'best_box': None,
                'best_conf': 0.0,
                'error': str(e),
            }
    
    def detect_and_crop(
        self,
        image: Union[Image.Image, str],
        padding: float = 0.05,
        return_metadata: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, Dict]]:
        """
        Detect object and crop image to best bounding box.
        
        Parameters
        ----------
        image : PIL.Image.Image or str
            Image to process. Can be PIL Image or path to image file.
        padding : float, default=0.05
            Padding to add around detected bbox (as fraction of bbox size).
        return_metadata : bool, default=False
            If True, return tuple of (cropped_image, metadata).
            
        Returns
        -------
        PIL.Image.Image or tuple
            Cropped image, or tuple of (cropped_image, detection_metadata)
            if return_metadata=True. Returns original image if no detection.
        """
        # Load image if path provided
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image
        
        # Detect objects
        detection = self.detect(img)
        
        if not detection['detected'] or detection['best_box'] is None:
            if return_metadata:
                return img, detection
            return img
        
        # Get bbox with padding
        x1, y1, x2, y2 = detection['best_box']
        width = x2 - x1
        height = y2 - y1
        
        # Add padding
        pad_x = width * padding
        pad_y = height * padding
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img.width, x2 + pad_x)
        y2 = min(img.height, y2 + pad_y)
        
        # Crop image
        cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))
        
        if return_metadata:
            detection['crop_box'] = [int(x1), int(y1), int(x2), int(y2)]
            return cropped, detection
        
        return cropped
    
    def detect_batch(
        self,
        images: List[Union[Image.Image, str, np.ndarray]],
        return_all: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in a batch of images (GPU-efficient).
        
        Parameters
        ----------
        images : list
            List of images (PIL Images, paths, or numpy arrays).
        return_all : bool, default=False
            If True, return all detections for each image.
            
        Returns
        -------
        list of dict
            List of detection results, one per image.
        """
        if not self.available or self.model is None:
            return [{'detected': False, 'num_detections': 0}] * len(images)
        
        try:
            # Run batch inference
            results = self.model.predict(
                images,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device,
            )
            
            # Process results for each image
            batch_results = []
            for result in results:
                if len(result.boxes) == 0:
                    batch_results.append({
                        'detected': False,
                        'num_detections': 0,
                        'best_box': None,
                        'best_conf': 0.0,
                    })
                    continue
                
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                # Find best detection
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                scores = confs * areas
                best_idx = np.argmax(scores)
                
                detection_result = {
                    'detected': True,
                    'num_detections': len(boxes),
                    'best_box': boxes[best_idx].tolist(),
                    'best_conf': float(confs[best_idx]),
                    'best_class': int(classes[best_idx]),
                }
                
                if return_all:
                    detection_result['all_boxes'] = boxes.tolist()
                    detection_result['all_confs'] = confs.tolist()
                    detection_result['all_classes'] = classes.tolist()
                
                batch_results.append(detection_result)
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Error during batch detection: {e}")
            return [{'detected': False, 'num_detections': 0, 'error': str(e)}] * len(images)


def detect_and_crop(
    image_path: str,
    model_path: str,
    output_path: Optional[str] = None,
    device: str = 'cpu',
    padding: float = 0.05,
) -> Optional[str]:
    """
    Simple function to detect and crop an image.
    
    Parameters
    ----------
    image_path : str
        Path to input image.
    model_path : str
        Path to YOLO model weights.
    output_path : str, optional
        Path to save cropped image. If None, overwrites input.
    device : str, default='cpu'
        Device for inference.
    padding : float, default=0.05
        Padding around detected bbox.
        
    Returns
    -------
    str or None
        Path to cropped image, or None if no detection.
    """
    detector = YOLODetector(model_path=model_path, device=device)
    cropped, metadata = detector.detect_and_crop(
        image_path,
        padding=padding,
        return_metadata=True
    )
    
    if not metadata['detected']:
        return None
    
    save_path = output_path or image_path
    cropped.save(save_path)
    return save_path
