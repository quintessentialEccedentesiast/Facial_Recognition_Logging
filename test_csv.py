import cv2
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import uuid  # For generating unique IDs
import csv
from datetime import datetime

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

# Tracking dictionary to keep track of existing faces
tracked_faces = {}

# Function to calculate the centroid of a bounding box
def get_centroid(x, y, w, h):
    return (x + w // 2, y + h // 2)

# Function to calculate Euclidean distance
def euclidean_distance(centroid1, centroid2):
    return np.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)

# Initialize CSV file for logging entry and exit times by class name
with open("face_log.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Class Name", "Event", "Timestamp"])

# Start the video loop
while True:
    # Capture frame-by-frame
    success, frame = cap.read()

    if not success:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # List to store all detected faces and their classifications
    all_faces = []

    # Detect frontal faces
    frontal_faces = frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # If frontal faces are detected, classify them
    for (x, y, w, h) in frontal_faces:
        face_img = frame[y:y+h, x:x+w]
        class_name, confidence_score = classify_face(face_img)
        centroid = get_centroid(x, y, w, h)
        all_faces.append({'class_name': class_name, 'confidence': confidence_score, 'coords': (x, y, w, h), 'centroid': centroid, 'color': (0, 255, 0)})

    # Detect left profile faces
    left_profile_faces = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # If left profile faces are detected, classify them
    for (x, y, w, h) in left_profile_faces:
        face_img = frame[y:y+h, x:x+w]
        class_name, confidence_score = classify_face(face_img)
        centroid = get_centroid(x, y, w, h)
        all_faces.append({'class_name': class_name, 'confidence': confidence_score, 'coords': (x, y, w, h), 'centroid': centroid, 'color': (255, 0, 0)})

    # Detect right profile faces by flipping the frame
    flipped_frame = cv2.flip(gray, 1)  # Flip the grayscale image
    right_profile_faces = profile_face_cascade.detectMultiScale(flipped_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # If right profile faces are detected, classify them
    for (x, y, w, h) in right_profile_faces:
        # Adjust the x-coordinate for the flipped frame
        x = frame.shape[1] - x - w
        face_img = frame[y:y+h, x:x+w]
        class_name, confidence_score = classify_face(face_img)
        centroid = get_centroid(x, y, w, h)
        all_faces.append({'class_name': class_name, 'confidence': confidence_score, 'coords': (x, y, w, h), 'centroid': centroid, 'color': (0, 0, 255)})

    # Track existing faces and assign new IDs to new faces
    updated_tracked_faces = {}

    for face in all_faces:
        centroid = face['centroid']
        assigned = False

        # Try to match with an existing tracked face
        for face_id, tracked_data in tracked_faces.items():
            tracked_centroid = tracked_data['centroid']
            distance = euclidean_distance(centroid, tracked_centroid)
            if distance < 50:  # Threshold distance to consider it the same face
                # Update tracked face data
                updated_tracked_faces[face_id] = tracked_data
                updated_tracked_faces[face_id].update(face)
                updated_tracked_faces[face_id]['frame_count'] += 1

                # Log entry if the face appears for the required number of frames
                if not tracked_data['logged'] and tracked_data['frame_count'] == 5:
                    with open("face_log.csv", mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([face['class_name'], "Entry", datetime.now()])
                    updated_tracked_faces[face_id]['logged'] = True

                assigned = True
                break

        # If no match was found, assign a new ID
        if not assigned:
            new_id = str(uuid.uuid4())  # Generate a unique ID
            face['frame_count'] = 1
            face['logged'] = False
            updated_tracked_faces[new_id] = face

    # Check for faces that have left
    for face_id, tracked_data in tracked_faces.items():
        if face_id not in updated_tracked_faces:
            # Log exit if the face was logged as entered before
            if tracked_data['logged']:
                with open("face_log.csv", mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([tracked_data['class_name'], "Exit", datetime.now()])

    # Update tracked faces with the new tracking information
    tracked_faces = updated_tracked_faces

    # Draw bounding boxes for tracked faces
    for face_id, face_data in tracked_faces.items():
        x, y, w, h = face_data['coords']
        color = face_data['color']
        confidence_score = face_data['confidence']
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"Class: {face_data['class_name']}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {round(confidence_score * 100, 2)}%", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the frame with the bounding boxes and predictions
    cv2.imshow('Live Camera Feed', frame)

    # Press 'q' to quit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
