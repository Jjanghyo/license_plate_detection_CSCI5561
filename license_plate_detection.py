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

from ultralytics import YOLO
import cv2
import pytesseract
import matplotlib.pyplot as plt
import os

############################################################
# 1. TRAIN YOLO LICENSE PLATE DETECTOR
############################################################

def train_yolo_detector():
    """Train YOLOv8/YOLOv9 model on custom license plate dataset."""

    # Example: YOLOv8n
    model = YOLO("yolov8n.pt")  # or yolov9c/yolov8s

    # IMPORTANT: update dataset.yaml path
    model.train(
        data="lp_dataset.yaml",  # your dataset config
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        device="cpu",  # or "cpu"
        name="lp_detector_v1"
    )

    return model


############################################################
# 2. INFERENCE: DETECT LICENSE PLATES
############################################################

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


############################################################
# 3. OCR (Tesseract or custom)
############################################################

def ocr_plate(img, box):
    """Crop plate from image and apply OCR."""

    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]

    # Preprocess
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # OCR
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(gray, config=config)
    text = ''.join([c for c in text if c.isalnum()])

    return text, gray


############################################################
# 4. TEST PIPELINE (DETECTION + OCR)
############################################################

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


############################################################
# 5. EXAMPLE USAGE
############################################################

if __name__ == "__main__":

    # 1. TRAIN (uncomment)
    train_yolo_detector()

    # 2. TEST
    # model = "runs/detect/lp_detector_v1/weights/best.pt"
    # outputs = test_pipeline(model, "car_sample.jpg")
    # print("Detected plate numbers:", outputs)

    pass
