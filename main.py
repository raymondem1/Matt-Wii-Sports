from cProfile import label
import tensorflow
from tensorflow import _keras_module
import keras
from tensorflow import keras
import numpy as np
import cv2
import keyboard
from process_labels import gen_labels
from collections import defaultdict
from playsound import playsound
import serial
import time

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
image = cv2.VideoCapture(2 + cv2.CAP_DSHOW)
ser = serial.Serial('COM4', 9600)

i=0
e=0
duh = 0
thing = 0
agh = 0
# Load the model
print("Starting")
model = tensorflow.keras.models.load_model("C:/Users/raymo/OneDrive/Desktop/shitty robot/elf - YT/keras_model.h5")
#model = tensorflow.keras.models.load_model("C:/Users/raymo/OneDrive/Desktop/shitty robot/elf - YT/k1.h5")

print("loaded")

"""
Create the array of the right shape to feed into the keras model
The 'length' or number of images you can put into the array is
determined by the first position in the shape tuple, in this case 1."""
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# A dict that stores the labels
labels = gen_labels()
print("rnning")
while True:
    # Choose a suitable font
    font = cv2.FONT_HERSHEY_SIMPLEX
    ret, frame = image.read()
    frame = cv2.flip(frame, 1)
    # In case the image is not read properly
    if not ret:
        continue
    # Draw a rectangle, in the frame
    frame = cv2.rectangle(frame, (220, 80), (530, 360), (0, 0, 255), 3)
    # Draw another rectangle in which the image to labelled is to be shown.
    frame2 = frame[80:360, 220:530]
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    frame2 = cv2.resize(frame2, (224, 224))
    # turn the image into a numpy array
    image_array = np.asarray(frame2)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    pred = model.predict(data)
    result = np.argmax(pred[0])
    #print(labels)
     #Print the predicted label into the screen.
    cv2.putText(frame,  "check : " +
                labels[str(result)], (280, 400), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


    # Exit, when 'q' is pressed on the keyboard
    if cv2.waitKey(1) and 0xff == ord('q'):
        exit = True
        break
    # Show the frame   

    #print(labels[str(0)])
    #print(labels[str(1)])
    #cv2.imshow('Frame', frame)
    #print(labels[str(result)])
    #print(result)

    if(result == 0):
        #print(result)
        if(thing ==0):
            ser.write(b'1')
            thing = 1
    if(result != 0):
        ser.write(b'0')
        thing = 0

image.release()
cv2.destroyAllWindows()
