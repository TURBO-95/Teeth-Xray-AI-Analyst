from ultralytics import YOLO
from flask import request, Flask, jsonify
from waitress import serve
from PIL import Image
import os

app = Flask(__name__)

# Load the model once when starting the server
MODEL_PATH = "C:/Users/youss/OneDrive/Desktop/AI-1/DentalXrayAI/best.pt"
try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    raise

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    try:
        template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
        with open(template_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading template: {e}")
        return f"Error loading page: {str(e)}", 500

@app.route("/detect", methods=["POST"])
def detect():
    """
    Handler of /detect POST endpoint
    Receives uploaded file with a name "image_file", passes it
    through YOLOv8 object detection network and returns array
    of bounding boxes.
    """
    try:
        if "image_file" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400
        
        image_file = request.files["image_file"]
        if image_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        boxes = detect_objects_on_image(image_file.stream)
        return jsonify(boxes)
    except Exception as e:
        print(f"Error in detect endpoint: {e}")
        return jsonify({"error": str(e)}), 500

def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    """
    try:
        # Open and convert image
        image = Image.open(buf)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Run prediction
        results = model.predict(image)
        result = results[0]
        output = []
        
        # Process detection results
        for box in result.boxes:
            x1, y1, x2, y2 = [
                round(x) for x in box.xyxy[0].tolist()
            ]
            class_id = box.cls[0].item()
            prob = round(box.conf[0].item(), 2)
            prob_percentage = f"{prob * 100:.2f}%"
            output.append([
                x1, y1, x2, y2, result.names[class_id], prob_percentage
            ])
        return output
    except Exception as e:
        print(f"Error during detection: {e}")
        return []

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8080)
