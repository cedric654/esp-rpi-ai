from flask import Flask, Response, render_template
import cv2
import numpy as np
import importlib.util

app = Flask(__name__)

# Load the TensorFlow Lite model
pkg = importlib.util.find_spec("tflite_runtime")
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

interpreter = Interpreter(
    model_path="model/weapons_detection_model_lite/detect.tflite"
)  # DEFINE MODEL HERE
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Configure the camera
camera = cv2.VideoCapture(0)  # Change the index if using a different camera
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Preprocess the input frame
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame = cv2.resize(
            input_frame, (input_details[0]["shape"][2], input_details[0]["shape"][1])
        )
        input_frame = np.expand_dims(input_frame, axis=0)
        input_frame = (
            input_frame.astype(np.float32) - 127.5
        ) / 127.5  # Normalize to [-1, 1]

        # Set the input tensor
        interpreter.set_tensor(input_details[0]["index"], input_frame)

        # Run inference
        interpreter.invoke()

        # Get the output tensors
        output_boxes = interpreter.get_tensor(output_details[0]["index"])
        output_classes = interpreter.get_tensor(output_details[1]["index"])
        output_scores = interpreter.get_tensor(output_details[2]["index"])
        num_detections = int(output_details[3]["index"][0])  # ERROR

        # Process the output detections
        for i in range(num_detections):
            if output_scores[0, i] > 0.5:  # Change the threshold as needed
                ymin, xmin, ymax, xmax = output_boxes[0, i]
                xmin = int(xmin * frame.shape[1])
                xmax = int(xmax * frame.shape[1])
                ymin = int(ymin * frame.shape[0])
                ymax = int(ymax * frame.shape[0])

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template(
        "index.html"
    )  # You can create an HTML template for the MJPEG viewer


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
