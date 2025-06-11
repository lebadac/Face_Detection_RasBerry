import time
import cv2
import requests
import base64
import numpy as np

# URL of the server for face recognition
url = 'http://127.0.0.1:8000/recog'  # Change to server IP if over LAN

# Function to encode an image to base64 string
def encode_image_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

# Open the webcam (device 0 by default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if frame cannot be read

    # Resize the frame to reduce size for faster processing
    resized = cv2.resize(frame, (320, 240))

    # Convert image to base64 for HTTP transmission
    encoded_image_data = encode_image_to_base64(resized)

    # Prepare the data to send to the server
    data = {
        'image': encoded_image_data,
        'w': 320,
        'h': 240
    }

    try:
        start_time = time.time()  # Start measuring time before sending

        # Send the image to the server for recognition
        response = requests.post(url, data=data, timeout=3)

        # Calculate the elapsed time after getting response
        elapsed_time = time.time() - start_time
        result = response.json()  # Parse the JSON response

        print(f"⏱️ Total recognition time: {elapsed_time * 1000:.2f} ms")

        name = result.get('name', 'Unknown')
        box = result.get('box', None)
        confidence = result.get('confidence', 0.0)

        if box:
            # Draw bounding box around the detected face
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display name and confidence
            label = f"{name} ({confidence * 100:.1f}%)"
            cv2.putText(resized, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the total time taken for recognition on the frame
        time_label = f"Time: {elapsed_time * 1000:.2f} ms"
        cv2.putText(resized, time_label, (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    except Exception as e:
        print("Error:", e)

    # Show the processed frame
    cv2.imshow("Face Recognition", resized)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
