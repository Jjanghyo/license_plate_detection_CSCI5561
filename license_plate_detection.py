# YOLO-based License Plate Detection + OCR Pipeline (Training + Testing Skeleton)

"""
This is a full pipeline skeleton for:
1. Training a YOLO model for license plate detection
2. Running inference (plate detection)
3. Cropping the plate and applying OCR

You will need:
- A labeled dataset in YOLO format (images + .txt label files)
- ultralytics library (pip install ultralytics)
- pytesseract or a deep OCR model

Fill in dataset paths and train parameters as needed.
"""

from paddleocr import PaddleOCR
from ultralytics import YOLO
import cv2
import pytesseract
import matplotlib.pyplot as plt

rec_engine = PaddleOCR(
    use_angle_cls=True,
    lang='en',
)

# 1. TRAIN YOLO LICENSE PLATE DETECTOR
def train_yolo_detector():
    """Train YOLOv8 model on custom license plate dataset."""

    model = YOLO("yolov8n.pt")

    # IMPORTANT: update dataset.yaml path
    model.train(
        data="lp_dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        device="cpu",  # or "cuda:0" for GPU
        name="lp_detector_v1"
    )

    return model

# 2. INFERENCE: DETECT LICENSE PLATES
def detect_plates(model_path, image_path):
    """Detect license plates using trained YOLO model."""

    model = YOLO(model_path)

    results = model(image_path)[0]

    img = cv2.imread(image_path)
    plates = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plates.append((x1, y1, x2, y2))

        # draw
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display detection
    plt.figure(figsize=(8, 8))
    plt.title("Detected Plates")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    return plates, img

# 3. OCR (Tesseract or custom)
def ocr_plate(img, box):
    """Crop plate and run PaddleOCR."""

    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]

    # PaddleOCR expects BGR numpy image directly
    result = rec_engine.predict(crop)

    if not result or not result[0]:
        return "", crop

    text = result[0]['rec_texts']  # recognized text only
    print("OCR:", text)

    return text, crop
# def ocr_plate(img, box):
#     """Crop plate from image and apply OCR."""

#     x1, y1, x2, y2 = box
#     crop = img[y1:y2, x1:x2]

#     # Preprocess
#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#     gray = cv2.bilateralFilter(gray, 16, 17, 17)

#     # OCR
#     config = "--psm 12 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#     text = pytesseract.image_to_string(gray, config=config)
#     print("text ::: " +  text)
#     text = ''.join([c for c in text if c.isalnum()])

#     return text, gray

# 4. TEST PIPELINE (DETECTION + OCR)
def test_pipeline(model_path, image_path):
    plates, img = detect_plates(model_path, image_path)

    results = []
    for box in plates:
        text, pre = ocr_plate(img, box)

        # Show OCR input
        plt.figure(figsize=(5, 5))
        plt.title("OCR Input")
        plt.imshow(pre, cmap="gray")
        plt.axis("off")
        plt.show()

        results.append(text)

    return results

if __name__ == "__main__":

    # 1. TRAIN (uncomment)
    # train_yolo_detector()

    # 2. TEST
    model = "runs/detect/lp_detector_v12/weights/best.pt"
    outputs = test_pipeline(model, "car_sample.jpg")
    print("Detected plate numbers:", outputs)

    pass
