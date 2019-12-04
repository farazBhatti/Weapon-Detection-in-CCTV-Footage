# USAGE
# python predict.py --image dataset/evaluation/Egg/3_137.jpg 

# import the necessary packages
from keras.models import load_model
from pyimagesearch import config
import numpy as np

import imutils
import cv2

model = load_model(config.MODEL_PATH)

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, image = cap.read()
    # load the input image and then clone it so we can draw on it later
    output = image.copy()
    output = imutils.resize(output, width=400)
    
    # our model was trained on RGB ordered images but OpenCV represents
    # images in BGR order, so swap the channels, and then resize to
    # 224x224 (the input dimensions for VGG16)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    
    # convert the image to a floating point data type and perform mean
    # subtraction
    image = image.astype("float32")
    mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
    image -= mean
    
    # load the trained model from disk
    print("[INFO] loading model...")
    
    # pass the image through the network to obtain our predictions
    preds = model.predict(np.expand_dims(image, axis=0))[0]
    i = np.argmax(preds)
    label = config.CLASSES[i]
    
    # draw the prediction on the output image
    text = "{}: {:.2f}%".format(label, preds[i] * 100)
    cv2.putText(output, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    	(0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('frame',output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()