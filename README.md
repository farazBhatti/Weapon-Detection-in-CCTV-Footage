# Weapon-Detection-in-CCTV-Footage
Repository contains implementation of weapon detection pipeline which consists of 2 independently trained object detection and one classification algorithm.


VGG_16 was trained using [PyimageSearch](https://www.pyimagesearch.com/2019/05/20/transfer-learning-with-keras-and-deep-learning/) tutorial on gun(~11000) images.

## Requirements
Python 3.6 or later

- numpy
- opencv-python
- torch >= 1.1.0













## Inference
__VGG__

images :
``` python3 predict_VGG.py --image test_imgs/gun.jpg ```

webcame : 
``` python3 webcam_VGG.py ```

Note: To run webcam_VGG.py make new folder, name it "output", download VGG weights for link below and paste it there


## Pre-Trained Model Weights
__YOLO Backend__
[weights](https://drive.google.com/open?id=1uTlyDWlnaqXcsKOktP5aH_zRDbfcDp-y).

__VGG weights for weapon classifcation__
[weights](https://drive.google.com/file/d/1IjEGxk9UbJLeK04EltVUCDv8_RxVn1GX/view?usp=sharing)

