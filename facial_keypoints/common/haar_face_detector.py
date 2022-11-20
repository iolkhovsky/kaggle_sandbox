import cv2
import numpy as np


class HaarFaceDetector:
    def __init__(self):
        self._config = 'haarcascade_frontalface_default.xml'
        self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + self._config)

    def detect(self, frame):
        assert isinstance(frame, np.ndarray)
        assert len(frame.shape) == 3 and frame.shape[-1] == 3
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self._cascade.detectMultiScale(gray, 1.3, 5)

    def __call__(self, *args, **kwargs):
        return self.detect(*args, **kwargs)

    def __str__(self):
        return "Haar feature based face detector"


if __name__ == "__main__":
    facedet = FaceDetector()
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        boxes = facedet.detect(img)
        for idx, (x, y, w, h) in enumerate(boxes):
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("camera", img)
        if cv2.waitKey(10) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()