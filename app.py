from flask import Flask, Response
from flask_cors import CORS
import cv2
import dlib

app = Flask(__name__)
CORS(app)

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cam = cv2.VideoCapture(0)


def detect_face_and_landmarks(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_frame)

    for face in faces:
        landmarks = landmark_predictor(frame, face)
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
    return frame


def gen_frames():
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame = detect_face_and_landmarks(frame)
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
