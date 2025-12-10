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
import json
import os


rec_engine = PaddleOCR(
    use_angle_cls=True,
    lang='en',
)
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
# contains_truth_substring(predicted_labels, ground_truth_labels):
#   input:
#       predicted_labels - OCR with paddle produces a list of strings that are predictions for the different
#           words/strings it can detect in a license plate which includes the state or other strings
#           that may be seen in registration stickers.
#       ground_truth_labels - the ground truth license plate number.
#   output:
#       a list containing any element in the predicted labels list from OCR
#           that contains a substring of minimum length 2 that could indicate that it is the license plate
#           number in order to ignore any predicted strings that aren't related to the number itself.
#   sources:
#       https://www.geeksforgeeks.org/python/python-remove-all-characters-except-letters-and-numbers/
############################################################
def contains_truth_substring(predicted_labels, ground_truth_label):
    # min_substring_length = 2
    predicted_labels_flat = predicted_labels[0]

    print("PREDICTED LIST: ", predicted_labels_flat) #debug
    processed_labels = [ ''.join(filter(str.isalnum, label)).upper() for label in predicted_labels_flat]
    # Original attempt:
    # for label in predicted_labels:
    #     processed_label = ''.join(filter(str.isalnum, label)).upper()
    #     processed_labels.append(processed_label)
    # matching_labels = []
    
    for label in processed_labels:
        for i in range(len(ground_truth_label)):
            for j in range(i+ 2, len(ground_truth_label) +1):
                substring = ground_truth_label[i:j]
                if substring in label:
                    print("MATCHING LABEL:", label)
                    return [label]
    print("NO MATCHING LABEL FOUND")
    return []


############################################################
# ocr_accuracy_evaluation():
#   input: 
#       model_path - path to pretrained model
#       ground_truth_dict - ground truth dictionary (using testing images provided as .jpeg/.jpgfiles)
#   output:
#       prints OCR accuracy and outputs a json file with the results.
#   description:
#       Currently finds accuracy of test images (how many license plate images are correct vs incorrect).
#   Sources:
#       https://www.geeksforgeeks.org/python/iterate-over-a-dictionary-in-python/ 
#       https://www.geeksforgeeks.org/python/reading-and-writing-json-to-a-file-in-python/
############################################################
def ocr_accuracy_evaluation(model_path, ground_truth_dict):
    correct = False
    number_correct_imgs = 0
    total_imgs = 0
    total_character_accuracy = 0.0 # to find the average accuracy per character
    results =[]

    # iterating through dictionary to get the test image pathways and ground_truth_labels:
    for img_path, ground_truth_label in ground_truth_dict.items():
        total_plate_chars = len(ground_truth_label)
        correct_plate_chars = 0
        # running YOLO + OCR for character recognition
        raw_prediction_list = test_pipeline(model_path, img_path)

        matching_labels = contains_truth_substring(raw_prediction_list, ground_truth_label)
        prediction = ''.join(matching_labels)
        print("MATCH LABELS: ", matching_labels)
        total_imgs += 1
        # print(predictions) Debugging -- 3 basic cases, predictions is empty (no plates detected), 1 plate, or multiple plates.
        # Currently focusing on 1 plate only.
        if len(prediction) == 0:
            # list is empty - no plates detected
            prediction = ""
            # total_imgs += 1
        
        # comparing prediction to ground truth
        if prediction == ground_truth_label:
            correct = True
            number_correct_imgs += 1
            correct_plate_chars += total_plate_chars # character accuracy would be 1.0

        else:
            # otherwise, correct remains as false and we need to cound total correct plate characters predicted
            for index, char in enumerate(prediction):
                #fixed bug: if the predicted string is longer than the ground_truth, just checking if the char is equal to the ground truth at
                # and index would break.
                if index < len(ground_truth_label) and char == ground_truth_label[index]:
                    correct_plate_chars += 1
            if len(prediction) > len(ground_truth_label):
                # deducting pionts if there are extra characters in the predicted label vs ground truth
                deduction = len(prediction) - len(ground_truth_label)
                correct_plate_chars -= deduction
            # was geting negative accuracies per character, so adding below
            if correct_plate_chars < 0:
                correct_plate_chars = 0

        if total_plate_chars == 0:
            character_accuracy = 0
        else:
            character_accuracy = correct_plate_chars /total_plate_chars
        total_character_accuracy = total_character_accuracy + character_accuracy
        results.append({"image" :img_path, "license plate" : ground_truth_label, "predicted plate": prediction, "correct": correct, "accuracy per character": character_accuracy})
        correct = False # retun back to false as needed

    # Finding accuracy in correct images vs incorrect images
    if total_imgs > 0:
        accuracy = number_correct_imgs / total_imgs
    else:
        accuracy =0
    print("OCR ACCURACY: ", accuracy)
    avg_character_accuracy = total_character_accuracy / total_imgs
    print("AVERAGE CHARACTER ACCURACY: ", avg_character_accuracy)
    # now saving to a JSON file
    json_results = {"accuracy" : accuracy, "average character accuracy": avg_character_accuracy, "total images" : total_imgs, "correct images": number_correct_imgs, "results": results}

    with open("ocr_accuracy_results.json", "w") as f:
        json.dump(json_results, f,indent=4)
    return json_results

############################################################
# ground_truth_plate_dict:
#   input: 
#       txt_file - takes a text file.
#   output:
#       ground_truth_plate_num - a dictionary where the keys are the paths to a test image and
#           values are the ground truth (true) license plate numbers found in the test images as
#           given by the text file.
#   Sources:
#       https://www.w3schools.com/python/ref_string_strip.asp
############################################################
def ground_truth_plate_dictionary(txt_file):
    ground_truth_plate_num = {}
    with open(txt_file, "r") as f:
        for line in f:
            img_path, plate_num = line.strip().split()
            ground_truth_plate_num[img_path]= plate_num
    return ground_truth_plate_num
############################################################
# 5. EXAMPLE USAGE
############################################################

if __name__ == "__main__":

    # 1. TRAIN (uncomment)
    # train_yolo_detector()

    # 2. TEST
    model = "runs/detect/lp_detector_v12/weights/best.pt"

    ground_truth_plate_nums = ground_truth_plate_dictionary("test_img.txt")
    ocr_accuracy_evaluation(model, ground_truth_plate_nums)
    # outputs = test_pipeline(model, "test_images/car_sample.jpeg")
    # print("Detected plate numbers:", outputs)


    pass
