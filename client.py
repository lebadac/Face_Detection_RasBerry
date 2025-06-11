import cv2
import requests
import base64
import numpy as np

url = 'http://127.0.0.1:8000/recog'  # thay bằng địa chỉ server nếu chạy qua mạng LAN

def encode_image_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không mở được webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame cho nhẹ
    resized = cv2.resize(frame, (320, 240))

    # Encode Base64
    encoded_image_data = encode_image_to_base64(resized)

    data = {
        'image': encoded_image_data,
        'w': 320,
        'h': 240
    }

    try:
        response = requests.post(url, data=data, timeout=3)
        result = response.json()
        name = result.get('name', 'Unknown')
        box = result.get('box', None)
        confidence = result.get('confidence', 0.0)

        if box:
            x1, y1, x2, y2 = [int(v) for v in box]
            # Vẽ bounding box
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Hiện tên + độ chính xác
            label = f"{name} ({confidence*100:.1f}%)"
            cv2.putText(resized, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    except Exception as e:
        print("Lỗi:", e)

    # Hiển thị frame
    cv2.imshow("Face Recognition", resized)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
