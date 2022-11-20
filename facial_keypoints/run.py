import argparse
import cv2
from copy import deepcopy
import yaml
import torch

from common.haar_face_detector import HaarFaceDetector
from model.keypoints_regressor import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Run keypoints detector on webcam stream')
    parser.add_argument('--model', type=str,
                        default='checkpoints/20-Nov-2022-19-27-31/state_dict_epoch_19_final')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model_config', type=str,
                        default='configs/deploy.yaml')
    return parser.parse_args()


def visualize_points(img, keypoints, origin=(0, 0), keypoint_sz=5, color=(0, 0, 255), thickness=2):
    y_org, x_org = origin
    for (y, x) in keypoints.reshape(-1, 2):
        img = cv2.circle(
            img,
            center=(int(x) + x_org, int(y) + y_org),
            radius=keypoint_sz // 2,
            color=color,
            thickness=thickness,
        )
    return img

def run(args):
    with open(args.model_config, 'rt') as f:
        config = yaml.safe_load(f.read())
    model = build_model(config)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    face_det = HaarFaceDetector()

    cap = cv2.VideoCapture(args.device)
    while True:
        ret, frame = cap.read()

        viz_frame = deepcopy(frame)
        boxes = face_det(frame)
        for x, y, w, h in boxes:
            viz_frame = cv2.rectangle(viz_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_subframe = frame[y:y+w, x:x+h, :]
            with torch.no_grad():
                keypoints = model(face_subframe)[0]
            viz_frame = visualize_points(viz_frame, keypoints.numpy(), origin=(y, x))

        cv2.imshow('frame', viz_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run(parse_args())
