import os
import cv2
import face_recognition

# Load reference images and their encodings from the "faces" folder
faces_folder = "faces"
reference_encodings = []
reference_names = []

# Iterate over each file in the faces folder
for filename in os.listdir(faces_folder):
    if filename.endswith(".png"):
        # Extract the name from the filename
        name = os.path.splitext(filename)[0]
        reference_names.append(name)

        # Load the image file
        image_path = os.path.join(faces_folder, filename)
        reference_image = face_recognition.load_image_file(image_path)

        # Encode the face in the reference image
        reference_encoding = face_recognition.face_encodings(reference_image)[0]
        reference_encodings.append(reference_encoding)

# Load the video capture
video_capture = cv2.VideoCapture(0)  # Replace with the video file path if you want to process a video file

# Adjust these parameters to optimize speed and accuracy
model = 'cnn'  # 'cnn' for faster but slightly less accurate model, 'hog' for more accurate but slower model
resize_factor = 0.5  # Resize factor for reducing image size (0.5 means half the original size)
frame_skip = 5  # Process every 5th frame

frame_counter = 0

while True:
    # Read a single frame from the video stream
    ret, frame = video_capture.read()

    frame_counter += 1
    if frame_counter % frame_skip != 0:
        continue

    # Resize the frame to improve performance
    small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

    # Convert the frame from BGR color (used by OpenCV) to RGB color (used by face_recognition)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame, model=model)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Iterate over each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the detected face encoding with the reference encodings
        matches = face_recognition.compare_faces(reference_encodings, face_encoding)

        # Check if any of the reference faces are detected
        if any(matches):
            # Get the names of the matched faces
            matched_names = [name for name, match in zip(reference_names, matches) if match]

            # Print the names of the detected faces and their location
            for name in matched_names:
                print(f"{name}'s face detected at coordinates: Top: {top}, Right: {right}, Bottom: {bottom}, Left: {left}")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()
