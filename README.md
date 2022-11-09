# Flag detection using YOLO
A project to identify the colour of a flag from a photograph. [YOLO](https://github.com/ultralytics/yolov5) has been used in the development of this project.

## Prerequisite

- Python3
- Flask
- [YOLOv5](https://github.com/ultralytics/yolov5)
- Object Detection

## Installation

### Please follow [My GuideBook](https://github.com/parth-p-7span/my-python-guidebook/blob/main/README.md) if you are new to Python programming.

1. Install Python3.x in your machine
2. Clone this repository in your workspace
```shell
git clone https://github.com/parth-p-7span/YOLO-flag-detection.git
```
3. Navigate to YOLO-flag-detection folder in your terminal/CMD
```shell
cd YOLO-flag-detection
```
4. Install required packages for the project using below command
```shell
pip install -r requirements.txt
```
5. To run the project open `yolov5/my_pred.py` file and set your testing image path in `image_path` variable
6. Run `my_pred.py` file

### To get the bounding boxed image
1. Open `yolov5/detect.py` file and set your testing image path in `image_path` variable
2. Run `detect.py` file.
3. You will get image with bounding box in `yolov5/runs/detect/` folder.

## File Structure

- `dataset/` : A folder containing dataset images and the corresponding labels for validation and training.
- `weights/` : It has model files that have been trained. Out of 50 epochs, `best.pt` is the model with the highest accuracy, and `last.pt` is the trained model at the very last epoch (in this case 50).
- `yolov5/` : The main repository for the Yolov5 model. It contains every file required to train, export, and test the model.
- `yolov5/train.py` : A python script to train custom object detection model using yolov5 model.
- `yolov5/detect.py` : A python script to predict outcomes from the pre-trained model.
- `yolov5/my_pred.py` : A python script to get the color of flag from the image.
- `yolov5/color_extractor.py` : A Python script with all the necessary functions to determine the flag's color.

## Important commands
- Train YOLOv5s on COCO128 for 3 epochs
```shell
python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt --cache
```

- Predict object from image using trained model
```shell
python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images
```
