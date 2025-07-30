# 🚗 Car Dent Detector

An AI-powered tool that detects and classifies dents on car images using computer vision. It automatically highlights damaged areas with bounding boxes and labels the dent type — streamlining vehicle inspections for car owners, mechanics, and insurance companies.

---

## 🛠️ Features

* Detects dents in car images
* Draws bounding boxes around damaged areas
* Classifies dent severity (e.g., minor, moderate, severe)
* Runs on a pre-trained PyTorch model (`best.pt`)

---

## 🚀 How to Run the Project

Follow these simple steps to get started:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/car-dent-detector.git
cd car-dent-detector
```

### 2. Download and Extract the Model

* Download the pre-trained model file: `best.pt.zip`
* Unzip it inside the project folder:

```bash
unzip best.pt.zip
```

Make sure `best.pt` is now in the root directory or in the correct model path used in the code.

### 3. Install Dependencies

Ensure you have Python 3.8+ and install the required packages:

```bash
!pip install ultralytics gradio opencv-python-headless scipy --quiet

from ultralytics import YOLO
import gradio as gr
import cv2
import numpy as np
from scipy.io.wavfile import write
from IPython.display import Audio, display
```


### 4. Run the Main Script

To detect dents in an image, simply run:

```bash
main.py --source path/to/your/image.jpg
```

The output image with detected dents will be saved in the `runs/` folder.

---

## 🧠 Model Info

* Framework: PyTorch
* Based on YOLOv8 for object detection
* Trained on a custom dataset of car dents

---

## 📂 Folder Structure

```
car-dent-detector/
│
├── main.py                  # Main detection script
├── best.pt                  # Pre-trained model (after unzipping)
├── requirements.txt         # Dependencies
├── runs/                    # Output images with results
└── README.md                # Project documentation
```

---

## 🤝 Contributing

Feel free to fork the repo, improve the model or UI, and submit a pull request!

---

## 📧 Contact

Have questions or suggestions? Open an issue or reach out at [saifullahrao089@gmail.com].
