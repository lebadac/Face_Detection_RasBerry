import time
import base64
import requests
from picamera2 import Picamera2
from PIL import Image
from io import BytesIO

# 🟢 Initialize camera once
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration({"size": (320, 240)}))
picam2.start()
time.sleep(2)  # Allow camera to stabilize

url = "http://172.31.9.179:8000/recog"  # Replace with your server IP

counter = 0  # 🔢 Frame counter

try:
    while True:
        print(f"\n📸 Frame #{counter + 1} - Capturing image...")

        # Capture image to array
        image_array = picam2.capture_array()

        # Convert to JPEG in memory
        image = Image.fromarray(image_array)
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        jpeg_data = buffer.getvalue()

        # (Optional) Save image locally
        filename = f"captured_{counter}.jpg"
        with open(filename, "wb") as f:
            f.write(jpeg_data)
        print(f"🖼️ Image saved: {filename}")

        # Encode to base64
        encoded_image = base64.b64encode(jpeg_data).decode('utf-8')

        # Prepare payload
        data = {
            "image": encoded_image,
            "w": 320,
            "h": 240
        }

        # Send request to server
        start = time.time()
        try:
            response = requests.post(url, data=data, timeout=5)
            elapsed = (time.time() - start) * 1000

            result = response.json()
            name = result.get("name", "Unknown")
            confidence = result.get("confidence", 0.0)

            print("📥 Server response:")
            print(f"🧑 Detected name: {name}")
            print(f"🎯 Accuracy: {confidence * 100:.2f}%")
            print(f"⏱️ Total processing time: {elapsed:.2f} ms")

        except Exception as e:
            print("❌ Error while sending image:", e)

        counter += 1
        time.sleep(5)  # ⏳ Wait 5 seconds before next loop

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
finally:
    picam2.stop()