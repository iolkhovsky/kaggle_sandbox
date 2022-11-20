import cv2
import numpy as np


class BaseDetector:

    def __init__(self):
        self._xml_config = None
        self._cascade = None
        self._build()
        return

    def _build(self):
        if self._xml_config is not None:
            self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + self._xml_config)
        return

    def detect(self, frame):
        if self._cascade is None:
            raise TypeError("Haar detector is not initialized")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self._cascade.detectMultiScale(frame, 1.3, 5)

    def __call__(self, *args, **kwargs):
        frame = None
        if len(args) != 0:
            frame = args[0]
        elif "frame" in kwargs.keys():
            frame = kwargs["frame"]
        if type(frame) != np.ndarray:
            raise TypeError("Missed frame argument")
        return self.detect(args[0])

    def __str__(self):
        return "Haar feature based detector"


class FaceDetector(BaseDetector):

    def __init__(self):
        self._xml_config = "haarcascade_frontalface_default.xml"
        self._build()
        return


class EyeDetector(BaseDetector):

    def __init__(self):
        self._xml_config = "haarcascade_eye.xml"
        self._build()
        return


if __name__ == "__main__":
    facedet = FaceDetector()
    eyedet = EyeDetector()
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        boxes = facedet.detect(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for idx, (x, y, w, h) in enumerate(boxes):
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            eyes = eyedet.detect(gray[y:y+h, x:x+w])
            for ex, ey, ew, eh in eyes:
                img = cv2.rectangle(img, (ex + x, ey + y), (ex + x + ew, ey + y + eh), (0, 255, 0), 2)
        cv2.imshow("camera", img)
        if cv2.waitKey(10) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()