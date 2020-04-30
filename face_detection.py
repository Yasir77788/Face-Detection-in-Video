# Face Detection from video
# Yasir Hassan
# Using python, opencv, and numpy

#import libraries
import cv2            # for opencv
import numpy as np    # for scientific computing with Python

# Load the cascade
# initialize the Haarcascade for frontal face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# The output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')


# import the input video
cap = cv2.VideoCapture('input_video.mp4') 

# the output video
output_movie =   cv2.VideoWriter('square_box_faces.avi',fourcc, 20.0,(int(cap.get(3)),int(cap.get(4))))

# capture frame by frame
while True:

    # Read the frame
    _, frame = cap.read()

    # Convert RGB to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           

    # Detect the faces by using face cascade
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        
        # using random for random colors of the rectangle labels
        color = list(np.random.random(size=3) * 256)

        # draw a bounding rectangle around each face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
      
    # Write the resulting image to the output video file
    output_movie.write(frame)
    
    # display the frame
    cv2.imshow('Detecting Faces', frame)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
# Release the VideoCapture object
cap.release()
