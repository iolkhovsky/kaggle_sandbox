import argparse
import cv2
import yaml
import torch

from model.keypoints_regressor import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Run keypoints detector on webcam stream')
    parser.add_argument('--model', type=str,
                        default='checkpoints/20-Nov-2022-19-27-31/state_dict_epoch_19_final')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model_config', type=str,
                        default='configs/deploy.yaml')
    return parser.parse_args()


def run(args):
    with open(args.model_config, 'rt') as f:
        config = yaml.safe_load(f.read())
    model = build_model(config)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    cap = cv2.VideoCapture(args.device)
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run(parse_args())
