from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset.sample import Sample


def plot_samples(samples, cols=4, cmap='gray', figsize=(16, 16)):
    sample_size = len(samples)
    rows = sample_size // cols
    if sample_size % cols:
        rows += 1
    fig, m_axs = plt.subplots(rows, cols, figsize=figsize)
    for (img, label), c_ax in zip(samples, m_axs.flatten()):
        c_ax.imshow(img, cmap=cmap)
        c_ax.set_title(f'{label}')
        c_ax.axis('off')


def visualize_sample(samples, cols=4, keypoint_sz=5, color=(255, 0, 0), thickness=1, figsize=(16, 16)):
    if isinstance(samples, Sample):
        samples = [samples]
    assert isinstance(samples, list), f"{type(sample)}"
    samples_amount = len(samples)
    assert samples_amount > 0, f"{samples_amount} < 1"
    viz_imgs, labels = [], []
    for sample in samples:
        labels.append(sample.hint)
        img = deepcopy(sample.image)
        for _, (y, x) in sample.keypoints.items():
            img = cv2.circle(
                img,
                center=(int(x), int(y)),
                radius=keypoint_sz // 2,
                color=color,
                thickness=thickness,
            )
        viz_imgs.append(img)
    return plot_samples(list(zip(viz_imgs, labels)), cols=cols, figsize=figsize)


def visualize_training(image_batch, pred_batch, gt_batch,
                       gt_color=(0, 255, 0),
                       pr_color=(0, 0, 255),
                       line_color=(255, 0, 0),
                       draw_lines=True,
                       keypoint_sz=5, thickness=1, figsize=(16, 16),
                       ret_images=False):
    if isinstance(image_batch, torch.Tensor):
        image_batch = image_batch.detach().numpy()
    if isinstance(pred_batch, torch.Tensor):
        pred_batch = pred_batch.detach().numpy()
    if isinstance(gt_batch, torch.Tensor):
        gt_batch = gt_batch.detach().numpy()

    viz_imgs, labels = [], []
    for idx, img in enumerate(image_batch):
        labels.append(f"Sample {idx}")
        viz_img = deepcopy(img)

        pred_points = pred_batch[idx].reshape(-1, 2)
        gt_points = gt_batch[idx].reshape(-1, 2)

        for pred_point, gt_point in zip(pred_points, gt_points):
            pred_y, pred_x = tuple(int(x) for x in pred_point)
            gt_y, gt_x = tuple(int(x) for x in gt_point)
            viz_img = cv2.circle(
                viz_img,
                center=(gt_x, gt_y),
                radius=keypoint_sz // 2,
                color=gt_color,
                thickness=thickness,
            )
            viz_img = cv2.circle(
                viz_img,
                center=(pred_x, pred_y),
                radius=keypoint_sz // 2,
                color=pr_color,
                thickness=thickness,
            )
            if draw_lines:
                viz_img = cv2.line(
                    viz_img,
                    (gt_x, gt_y),
                    (pred_x, pred_y),
                    line_color,
                    thickness
                )
        viz_imgs.append(viz_img)
    if ret_images:
        return viz_imgs
    else:
        plot_samples(list(zip(viz_imgs, labels)), cols=max(1, len(image_batch) // 2), figsize=figsize)