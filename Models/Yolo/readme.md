# How to train YOLO using ultralytics

## Requirements
- python 3.11
- ultralytics
- dataset (custom, optional)

## Process
1. Create virtual environment & activate
```bash
python3.11 -m venv yolo_training_env
source yolo_training_env/bin/activate
```

2. Install ultralytics
```bash
pip install ultralytics
```

3. Train YOLO (using custom dataset)
- Place the file structured as the following:
```
dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

- Place images under the "images" directory, and their descriptions (in txt) under "labels" directory
- txt format (for detection):
    ```<class_id> <x_center> <y_center> <width> <height>```

- After dataset setup is complete, run the training file:
```bash
python train.py
```

- The following files/directories will be created after training:
```
yolo_training/
├── yolo_run_v1/
│   ├── weights/
│   │   ├── best & last .pt
│   │   └── best.onxx (due to export function in the script)
│   ├── args.yaml
│   ├── results (.csv, .png)
│   └── visual evaluation files (.png)
├── yolo11n.pt
└── yolov8n.pt
```

4. Convert the .pt model to .onnx
- done in training script function ```model.export```