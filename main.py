import cv2
import torch
import torchvision.transforms as T
from facenet_pytorch import MTCNN

mtcnn = MTCNN(keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu") # multi-task cascaded cnn

cam = cv2.VideoCapture(0)

clahe = cv2.createCLAHE(
    clipLimit=2.0, tileGridSize=(8, 8)
)  # contrast limited adaptive histogram equalization

while True:
    ret, frame = cam.read()
    if not ret:
        break

    yCbCr_frame = cv2.cvtColor(
        frame, cv2.COLOR_BGR2YCrCb
    )  # convert frame from BGR to YCbCr color space.

    yCbCr_frame[:, :, 0] = clahe.apply(
        yCbCr_frame[:, :, 0]
    )  # apply CLAHE to the luma component (Y channel) of the YCbCr image.

    clahe_frame = cv2.cvtColor(
        yCbCr_frame, cv2.COLOR_YCrCb2BGR
    )  # convert the YCbCr image back to BGR color space.

    rgb_frame = cv2.cvtColor(
        clahe_frame, cv2.COLOR_BGR2RGB
    )  # convert frame from BGR to RGB

    boxes, probs = mtcnn.detect(rgb_frame)  # detect faces

    if boxes is not None:
        for box, prob in zip(boxes, probs):
            if prob > 0.90:  # only consider faces with confidence > 90%
                x1, y1, x2, y2 = [int(coord) for coord in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Face: {prob:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (36, 255, 12),
                    1,
                )

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
