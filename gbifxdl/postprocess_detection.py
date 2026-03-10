# Post-processing script for applying OCR and YOLO detection to already downloaded images.
# This script can be used to apply OCR text detection and/or YOLO object detection
# to images that have already been downloaded, without needing to re-download them.

import argparse
import logging
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from PIL import Image

from gbifxdl.ocr_detector import OCRDetector
from gbifxdl.yolo_detector import YOLODetector


def setup_logger(log_dir: Path) -> logging.Logger:
    """Set up logging."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = log_dir / "postprocess_detection.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def postprocess_with_detection(
    parquet_path: str,
    img_dir: str,
    output_parquet_path: Optional[str] = None,
    # OCR parameters
    use_ocr: bool = False,
    exclude_text_images: bool = True,
    ocr_confidence: float = 60.0,
    ocr_min_text_length: int = 3,
    # YOLO parameters
    use_yolo: bool = False,
    yolo_model_path: Optional[str] = None,
    yolo_model_repo: Optional[str] = None,
    yolo_model_filename: Optional[str] = None,
    yolo_device: str = 'cpu',
    yolo_conf_threshold: float = 0.25,
    yolo_padding: float = 0.05,
    batch_size: int = 1000,
    overwrite_images: bool = False,
):
    """
    Apply OCR and/or YOLO detection to already downloaded images.
    
    Parameters
    ----------
    parquet_path : str
        Path to the parquet file containing image metadata.
    img_dir : str
        Directory containing downloaded images.
    output_parquet_path : str, optional
        Path for output parquet file. If None, adds "_detected" suffix.
    use_ocr : bool, default=False
        Whether to apply OCR text detection.
    exclude_text_images : bool, default=True
        If True, remove images with detected text.
    ocr_confidence : float, default=60.0
        OCR confidence threshold.
    ocr_min_text_length : int, default=3
        Minimum text length for OCR detection.
    use_yolo : bool, default=False
        Whether to apply YOLO object detection and cropping.
    yolo_model_path : str, optional
        Path to YOLO model weights.
    yolo_model_repo : str, optional
        Hugging Face repo ID for YOLO model.
    yolo_model_filename : str, optional
        Filename of YOLO model in HF repo.
    yolo_device : str, default='cpu'
        Device for YOLO inference.
    yolo_conf_threshold : float, default=0.25
        YOLO confidence threshold.
    yolo_padding : float, default=0.05
        Padding around YOLO detections.
    batch_size : int, default=1000
        Batch size for processing parquet file.
    overwrite_images : bool, default=False
        Whether to overwrite original images with cropped versions.
    """
    
    parquet_path = Path(parquet_path)
    img_dir = Path(img_dir)
    
    logger = setup_logger(parquet_path.parent)
    logger.info(f"Starting post-processing detection on {parquet_path}")
    logger.info(f"OCR enabled: {use_ocr}, YOLO enabled: {use_yolo}")
    
    # Initialize detectors
    ocr_detector = None
    if use_ocr:
        logger.info("Initializing OCR detector...")
        ocr_detector = OCRDetector(
            confidence_threshold=ocr_confidence,
            min_text_length=ocr_min_text_length,
            logger=logger
        )
    
    yolo_detector = None
    if use_yolo:
        logger.info(f"Initializing YOLO detector on {yolo_device}...")
        yolo_detector = YOLODetector(
            model_path=yolo_model_path,
            model_repo=yolo_model_repo,
            model_filename=yolo_model_filename,
            device=yolo_device,
            conf_threshold=yolo_conf_threshold,
            logger=logger
        )
    
    # Set up output path
    if output_parquet_path is None:
        output_parquet_path = parquet_path.with_stem(parquet_path.stem + "_detected")
    else:
        output_parquet_path = Path(output_parquet_path)
    
    # Process parquet file in batches
    parquet_file = pq.ParquetFile(parquet_path)
    writer = None
    
    total_processed = 0
    total_excluded = 0
    total_cropped = 0
    
    for batch in tqdm(parquet_file.iter_batches(batch_size=batch_size), desc="Processing batches"):
        batch_df = batch.to_pandas()
        
        # Process each row
        for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Processing images", leave=False):
            if pd.isna(row.get('filename')) or row.get('filename') == '':
                continue
            
            img_path = img_dir / str(row['speciesKey']) / row['filename']
            
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
            
            try:
                with Image.open(img_path) as img:
                    ocr_metadata = {}
                    yolo_metadata = {}
                    should_exclude = False
                    should_crop = False
                    cropped_img = None
                    
                    # Apply OCR
                    if use_ocr and ocr_detector is not None:
                        ocr_result = ocr_detector.detect_text(img)
                        ocr_metadata = {
                            'has_text': ocr_result.get('has_text', False),
                            'text_confidence': ocr_result.get('text_confidence', 0.0),
                            'num_words': ocr_result.get('num_words', 0),
                        }
                        
                        if exclude_text_images and ocr_result.get('has_text', False):
                            should_exclude = True
                            logger.debug(f"Excluding {img_path} due to text detection")
                    
                    # Apply YOLO
                    if use_yolo and yolo_detector is not None and not should_exclude:
                        detection = yolo_detector.detect(img)
                        
                        if detection.get('detected', False):
                            # Crop to best bounding box
                            x1, y1, x2, y2 = detection['best_box']
                            width = x2 - x1
                            height = y2 - y1
                            
                            # Add padding
                            pad_x = width * yolo_padding
                            pad_y = height * yolo_padding
                            
                            x1 = max(0, x1 - pad_x)
                            y1 = max(0, y1 - pad_y)
                            x2 = min(img.width, x2 + pad_x)
                            y2 = min(img.height, y2 + pad_y)
                            
                            cropped_img = img.crop((int(x1), int(y1), int(x2), int(y2)))
                            should_crop = True
                            
                            yolo_metadata = {
                                'yolo_detected': True,
                                'yolo_confidence': detection.get('best_conf', 0.0),
                                'yolo_class': detection.get('best_class', -1),
                                'yolo_bbox': str(detection['best_box']),
                            }
                        else:
                            yolo_metadata = {
                                'yolo_detected': False,
                                'yolo_confidence': 0.0,
                                'yolo_class': -1,
                            }
                    
                    # Handle exclusion or cropping
                    if should_exclude:
                        os.remove(img_path)
                        batch_df.at[idx, 'status'] = 'excluded_text_detected'
                        batch_df.at[idx, 'filename'] = ''
                        total_excluded += 1
                    elif should_crop and cropped_img is not None:
                        if overwrite_images:
                            cropped_img.save(img_path)
                            batch_df.at[idx, 'width'] = cropped_img.width
                            batch_df.at[idx, 'height'] = cropped_img.height
                            total_cropped += 1
                    
                    # Update metadata
                    for key, value in {**ocr_metadata, **yolo_metadata}.items():
                        batch_df.at[idx, key] = value
                    
                    total_processed += 1
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        # Write batch to output parquet
        batch_table = pa.Table.from_pandas(batch_df)
        
        if writer is None:
            writer = pq.ParquetWriter(output_parquet_path, batch_table.schema)
        
        writer.write_table(batch_table)
    
    if writer:
        writer.close()
    
    logger.info(f"Post-processing complete!")
    logger.info(f"Total processed: {total_processed}")
    logger.info(f"Total excluded: {total_excluded}")
    logger.info(f"Total cropped: {total_cropped}")
    logger.info(f"Output saved to: {output_parquet_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Post-process images with OCR and/or YOLO detection"
    )
    
    parser.add_argument(
        "--parquet", "-p",
        required=True,
        help="Path to parquet file with image metadata"
    )
    parser.add_argument(
        "--images", "-i",
        required=True,
        help="Directory containing images"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output parquet path (default: adds _detected suffix)"
    )
    
    # OCR arguments
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Enable OCR text detection"
    )
    parser.add_argument(
        "--exclude-text",
        action="store_true",
        default=True,
        help="Exclude images with detected text"
    )
    parser.add_argument(
        "--ocr-confidence",
        type=float,
        default=60.0,
        help="OCR confidence threshold (0-100)"
    )
    
    # YOLO arguments
    parser.add_argument(
        "--use-yolo",
        action="store_true",
        help="Enable YOLO object detection and cropping"
    )
    parser.add_argument(
        "--yolo-model",
        help="Path to YOLO model weights file"
    )
    parser.add_argument(
        "--yolo-repo",
        help="Hugging Face repo ID for YOLO model"
    )
    parser.add_argument(
        "--yolo-filename",
        help="Filename of YOLO model in HF repo"
    )
    parser.add_argument(
        "--yolo-device",
        default="cpu",
        help="Device for YOLO inference (cpu, cuda, cuda:0, etc.)"
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold (0-1)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original images with cropped versions"
    )
    
    args = parser.parse_args()
    
    postprocess_with_detection(
        parquet_path=args.parquet,
        img_dir=args.images,
        output_parquet_path=args.output,
        use_ocr=args.use_ocr,
        exclude_text_images=args.exclude_text,
        ocr_confidence=args.ocr_confidence,
        use_yolo=args.use_yolo,
        yolo_model_path=args.yolo_model,
        yolo_model_repo=args.yolo_repo,
        yolo_model_filename=args.yolo_filename,
        yolo_device=args.yolo_device,
        yolo_conf_threshold=args.yolo_conf,
        overwrite_images=args.overwrite,
    )


if __name__ == "__main__":
    main()
