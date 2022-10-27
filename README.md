# Flag detection using YOLO
A project to identify the colour of a flag from a photograph. [YOLO](https://github.com/ultralytics/yolov5) has been used in the development of this project.

## Installation

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
