import cv2
import os

# Video capture from webcam
video = cv2.VideoCapture(0)

# Load Haar Cascades for frontal and side profile face detection
frontal_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
side_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

count = 0

# Create a directory for storing images
nameID = str(input("Enter Your Name: ")).lower()
path = 'images2/' + nameID

isExist = os.path.exists(path)

if isExist:
    print("Name Already Taken")
    nameID = str(input("Enter Your Name Again: "))
else:
    os.makedirs(path)

while True:
    ret, frame = video.read()

    # Detect frontal faces
    frontal_faces = frontal_face_cascade.detectMultiScale(frame, 1.3, 5)

    # Detect left profile faces
    side_faces = side_face_cascade.detectMultiScale(frame, 1.3, 5)

    # Detect right profile faces by flipping the frame
    flipped_frame = cv2.flip(frame, 1)
    right_faces = side_face_cascade.detectMultiScale(flipped_frame, 1.3, 5)

    # Draw rectangles and capture frontal faces
    for (x, y, w, h) in frontal_faces:
        count += 1
        name = './images2/' + nameID + '/' + str(count) + '.jpg'
        print(f"Creating Images (Frontal Face): {name}")
        cv2.imwrite(name, frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # Draw rectangles and capture left profile faces
    for (x, y, w, h) in side_faces:
        count += 1
        name = './images2/' + nameID + '/' + str(count) + '.jpg'
        print(f"Creating Images (Left Side Face): {name}")
        cv2.imwrite(name, frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

    # Draw rectangles and capture right profile faces from the flipped frame
    for (x, y, w, h) in right_faces:
        # Adjust the x-coordinate because we flipped the frame horizontally
        x = frame.shape[1] - x - w
        count += 1
        name = './images2/' + nameID + '/' + str(count) + '.jpg'
        print(f"Creating Images (Right Side Face): {name}")
        cv2.imwrite(name, frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

    # Show the video feed with rectangles drawn
    cv2.imshow("WindowFrame", frame)

    # Break the loop after capturing 500 images
    if count > 1000:
        break

    cv2.waitKey(1)

video.release()
cv2.destroyAllWindows()
