# Face Mask Detection

This project implements a face mask detection system using the YOLOv8 object detection framework. It classifies faces in images, videos, or webcam feed as either with mask or without mask. The project also supports batch inference on uploaded images.



## Table of Contents

- [Features](#features)  
- [Dataset Preparation](#dataset-preparation)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Training](#training)  
- [Validation](#validation)  
- [Inference](#inference)  
- [Webcam Detection](#webcam-detection)  
- [Batch Image Detection](#batch-image-detection)  
- [License](#license)  


## 📌 Project Overview
This project trains a YOLOv8 model to classify whether a person is:
 with_mask
 without_mask

The system works on images, videos, real-time webcam streams, and multiple uploaded images.

It is designed to work with datasets that contain XML annotations (Pascal VOC format) and converts them into **YOLO format** automatically.


## Features

- Automatically converts XML annotations into YOLO format.
- Splits the dataset into training and validation sets.
- Training a YOLOv8 model for mask detection.
- Improves model performance using standard metrics.
- Real-time detection with a webcam.
- Batch detection for uploaded images with annotated output.

## Dataset Preparation

### 1. Dataset Format
1. Organize the dataset :

```
Dataset/
images/
  image1.png
  image2.png
   ...
annotations/
   image1.xml
   image2.xml
    ...
```

2. The script automatically converts XML annotations to YOLO `.txt` format and splits the dataset into training (80%) and validation (20%) sets.

---
### 2. Conversion & Splitting
The notebook automatically:
 reads all XML files
 extracts bounding boxes
 converts to YOLO format
 saves `.txt` files
 copies images to train/val folders
 creates `mask.yaml`

## Project Structure

```
Dataset/                      # Original images and XML annotations
 yolo_dataset/
 images/
    train/
      val/
    labels/
     train/
      val/
  mask.yaml                     # YOLOv8 dataset configuration file
   train_mask_yolo.ipynb         # Main training and inference notebook
   README.md
```

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_folder>
```

2. Install dependencies:

```bash
pip install ultralytics opencv-python ipywidgets pillow scikit-learn
```

3. Enable Jupyter widgets if using Jupyter Notebook:

```bash
jupyter nbextension enable --py widgetsnbextension
```

## 🔄 XML to YOLO Conversion
The XML converter:
- reads `<object>` tags from XML
- extracts xmin, ymin, xmax, ymax
- normalizes coords between 0–1
- writes YOLO labels as:
```
class x_center y_center width height
```
Example label:
```
0 0.453 0.621 0.122 0.200
```

## Training

1. Load the YOLOv8 pretrained model:

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
```

2. Train on your dataset:

```python
results = model.train(
    data="mask.yaml",
    epochs=30,
    batch=8,
    imgsz=640,
    device="cpu"
)
```


## Validation

After training, evaluate the model on the validation set:

```python
metrics = model.val()
print(metrics)
```

Metrics include precision, recall, and mAP for both classes: `with_mask` and `without_mask`.


## Inference

### Single Image

```python
image_path = "path_to_image.png"
results = model(image_path)
results.show()
```

### Webcam Detection

```python
import cv2
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow("Mask Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Batch Image Detection (with Upload Widget)

- Upload multiple images via the widget.  
- Annotated images are displayed and saved automatically:

```python
from ipywidgets import FileUpload
upload_widget = FileUpload(accept='image/', multiple=True)
display(upload_widget)

# Running detect_mask(upload_widget) to process images
```

Annotated images are saved in specified output directory.


## 🔁 Reproducing All Key Results
To fully reproduce the project:
1. Clone the repository
2. Install dependencies
3. Place your XML/images in `Dataset/`
4. Run the notebook
5. Train the model
6. Validate
7. Run inference or webcam detection

All commands and code are provided step-by-step.

## License

This project is released under the MIT License.  


## Notes

- Ensure that all images and XML annotations have matching filenames.
- Change `device="cpu"` to `device="0"` to train on GPU for faster results.
- The notebook allows for flexibility in both training and inference pipelines.

 
Date: December 2025

