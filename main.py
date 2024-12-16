import cv2
import dlib

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray_frame)

    for face in faces:
        x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())

        landmarks = landmark_predictor(frame, face)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow("Face & Landmark Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
