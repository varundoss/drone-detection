import cv2
from core import Core

def main():
    core = Core()  # Initialize core detection class
    video_capture = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with a video file path

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        resized_image, scale = core.pre_process_image(image)  # Pre-process the image

        boxes, scores, labels = core.predict(core.model, resized_image, scale)  # Predict
        detections = core.draw_boxes_in_image(frame, boxes, scores)  # Draw boxes

        cv2.imshow("Drone Detection", frame)  # Display the result

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
