# Weapon-Detection-in-CCTV-Footage
Repository contains implementation of weapon detection pipeline which consists of 2 independently trained object detection and one classification algorithm.


VGG_16 was trained using [PyimageSearch](https://www.pyimagesearch.com/2019/05/20/transfer-learning-with-keras-and-deep-learning/) tutorial on gun(~11000) images.

## Inference
__VGG__

images :
``` python3 predict_VGG.py --image test_imgs/gun.jpg ```

webcame : 
``` python3 webcam_VGG.py ```

## Pre-Trained Model Weights
__YOLO Backend__
[weights](https://drive.google.com/open?id=1uTlyDWlnaqXcsKOktP5aH_zRDbfcDp-y).


