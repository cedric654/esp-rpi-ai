"""
Project: AI Robot - Object Detection
Author: Jitesh Saini
Github: https://github.com/jiteshsaini
website: https://helloworld.co.in

The code does following:-
- The robot uses PiCamera to capture frames. 
- An object within the frame is detected using Machine Learning moldel & TensorFlow Lite interpreter. 
- Using OpenCV, the frame is overlayed with information such as: color coded bounding boxes, information bar to show FPS, Processing durations and an Object Counter.
- Stream the output window (camera view with overlays) over LAN through FLASK.
"""
import util as cm
import cv2
import numpy as np
from PIL import Image

cap = cv2.VideoCapture(0)
threshold = 0.2
top_k = 5  # number of objects to be shown as detected

model_dir = "model/weapons_detection_model_lite"
model = "detect.tflite"
lbl = "labelmap.txt"

prev_val = 0

selected_obj = ""

# ---------Flask----------------------------------------
from flask import Flask, Response, render_template

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    # global cap
    return Response(main(), mimetype="multipart/x-mixed-replace; boundary=frame")


# -------------------------------------------------------------
def main():
    mdl = model

    interpreter, labels = cm.load_model(model_dir, mdl, lbl)

    arr_dur = [0, 0, 0]
    # while cap.isOpened():
    while True:
        # ----------------Capture Camera Frame-----------------
        ret, frame = cap.read()
        if not ret:
            break

        cv2_im = frame
        cv2_im = cv2.flip(cv2_im, 0)
        cv2_im = cv2.flip(cv2_im, 1)

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)

        # -------------------Inference---------------------------------
        cm.set_input(interpreter, pil_im)
        interpreter.invoke()
        objs = cm.get_output(interpreter, score_threshold=threshold, top_k=top_k)

        # -----------------other------------------------------------
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cv2_im = cm.append_text_img1(cv2_im, objs, labels, arr_dur, selected_obj)
        # cv2.imshow('Object Detection - TensorFlow Lite', cv2_im)

        ret, jpeg = cv2.imencode(".jpg", cv2_im)
        pic = jpeg.tobytes()

        # Flask streaming
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + pic + b"\r\n\r\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2204, threaded=True)  # Run FLASK
    main()
