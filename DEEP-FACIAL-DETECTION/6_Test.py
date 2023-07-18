import cv2
from random import randrange
from keras.models import load_model

# Loading some pre-trained data
trained_face_data = load_model('facetracker.h5')

# Capture Video
webcam = cv2.VideoCapture(0)

# Iterate Forever over the frames
while True:
    # Read current frame
    successful_frame_read, frame = webcam.read(0)

    # Convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Let's detect all the faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw Rectangles
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(255), randrange(255), randrange(255)), 2)

    # Show the frame
    cv2.imshow('Face Detector', frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
webcam.release()

# Destroy all windows
cv2.destroyAllWindows()