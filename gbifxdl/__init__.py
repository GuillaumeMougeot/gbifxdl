from .gbifxdl import *
try:
    from .crop_img import *
except ImportError as e:
    print(
        "Some dependencies are missing, "
        "but gbifxdl core functionalities are operational. "
        f"Details: {e}"
    )

try:
    from .ocr_detector import *
except ImportError as e:
    print(
        "OCR dependencies missing (pytesseract). "
        "OCR functionality will not be available. "
        f"Details: {e}"
    )

try:
    from .yolo_detector import *
except ImportError as e:
    print(
        "YOLO dependencies missing (ultralytics). "
        "YOLO detection functionality will not be available. "
        f"Details: {e}"
    )
