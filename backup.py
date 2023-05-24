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
    model_path="model/weapons_detection_model_lite/detect.tflite",
)  # DEFINE MODEL HERE
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]["shape"][1]
width = input_details[0]["shape"][2]

floating_model = input_details[0]["dtype"] == np.float32

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]["name"]

if "StatefulPartitionedCall" in outname:  # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2


# Configure the camera
camera = cv2.VideoCapture(0)  # Change the index if using a different camera
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def generate_frames():
    while True:
        success, frame1 = camera.read()
        if not success:
            break

        # # Preprocess the input frame
        # input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # input_frame = cv2.resize(
        #     input_frame, (input_details[0]["shape"][2], input_details[0]["shape"][1])
        # )
        # input_frame = np.expand_dims(input_frame, axis=0)
        # input_frame = (
        #     input_frame.astype(np.float32) - 127.5
        # ) / 127.5  # Normalize to [-1, 1]

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]["index"], input_data)

        # Run inference
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]["index"])[
            0
        ]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]["index"])[
            0
        ]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]["index"])[
            0
        ]  # Confidence of detected objects

        # Process the output detections
        for i in range(num_detections):
            if output_scores[i] > 0.5:  # Change the threshold as needed
                ymin, xmin, ymax, xmax = output_boxes[i]
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
