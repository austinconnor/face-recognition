import cv2

def save_image(name, image):
    cv2.imwrite(f"faces/{name}.png", image)
    print(f"Image saved as {name}.png")

def train():
    # Prompt the user to enter a name
    name = input("Enter a name: ")

    # Load the webcam capture
    video_capture = cv2.VideoCapture(0)

    while True:
        # Read a single frame from the video stream
        ret, frame = video_capture.read()

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Break the loop if the spacebar is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            save_image(name, frame)
            break

    # Release the video capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    train()
