import cv2
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the pre-trained model
model = load_model("keras_Model.h5", compile=False)

# Load the class names from labels.txt
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Initialize the video capture object (0 for the default camera)
cap = cv2.VideoCapture(0)

# Set the desired camera resolution (Optional)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load the Haar Cascades for frontal and profile face detection
frontal_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profile_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

# Function to preprocess the frame and get predictions
def classify_face(face_img):
    # Convert the face (BGR from OpenCV) to RGB
    image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

    # Resize the image to 224x224 pixels (as required by the model)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Convert the image to a numpy array
    image_array = np.asarray(image)

    # Normalize the image to match the input range the model expects
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Prepare the array for the model (batch size of 1, 224x224x3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make the prediction
    prediction = model.predict(data)

    # Get the predicted class index and confidence score
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]

    return class_names[index], confidence_score

# Start the video loop
while True:
    # Capture frame-by-frame
    success, frame = cap.read()

    if not success:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_detected = False  # Flag to track if a face is detected

    # First, try to detect frontal faces
    frontal_faces = frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(frontal_faces) > 0:
        # If a frontal face is detected, classify and display it
        for (x, y, w, h) in frontal_faces:
            face_img = frame[y:y+h, x:x+w]
            class_name, confidence_score = classify_face(face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green for frontal
            cv2.putText(frame, f"Class: {class_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {round(confidence_score * 100, 2)}%", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            face_detected = True  # Mark that a face is detected
            break  # Stop further processing once a frontal face is found

    # If no frontal face is detected, check for left profile
    if not face_detected:
        left_profile_faces = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        if len(left_profile_faces) > 0:
            # Classify and display the left profile face
            for (x, y, w, h) in left_profile_faces:
                face_img = frame[y:y+h, x:x+w]
                class_name, confidence_score = classify_face(face_img)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue for left profile
                cv2.putText(frame, f"Class: {class_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, f"Confidence: {round(confidence_score * 100, 2)}%", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                face_detected = True
                break  # Stop further processing once a left profile face is found

    # If no frontal or left profile face is detected, check for right profile by flipping the frame
    if not face_detected:
        flipped_frame = cv2.flip(gray, 1)  # Flip the grayscale image
        right_profile_faces = profile_face_cascade.detectMultiScale(flipped_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(right_profile_faces) > 0:
            for (x, y, w, h) in right_profile_faces:
                # Adjust the x-coordinate for the flipped frame
                x = frame.shape[1] - x - w
                face_img = frame[y:y+h, x:x+w]
                class_name, confidence_score = classify_face(face_img)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red for right profile
                cv2.putText(frame, f"Class: {class_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, f"Confidence: {round(confidence_score * 100, 2)}%", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                face_detected = True
                break  # Stop further processing once a right profile face is found

    # Display the frame with the bounding boxes and predictions
    cv2.imshow('Live Camera Feed', frame)

    # Press 'q' to quit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()