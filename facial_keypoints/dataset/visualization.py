from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile
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


def visualize_distributions(pred_batch, gt_batch, figsize=(8, 4), img_size=(96, 96)):
    if isinstance(pred_batch, torch.Tensor):
        pred_batch = pred_batch.detach().numpy()
    if isinstance(gt_batch, torch.Tensor):
        gt_batch = gt_batch.detach().numpy()

    fig, m_axs = plt.subplots(1, 2, figsize=figsize)
    pr_ax = m_axs.flatten()[0]
    gt_ax = m_axs.flatten()[1]
    img_height, img_width = img_size

    def plot_hist(ax, batch, label):
        points = []
        for img_points in batch:
            img_points = img_points.reshape(-1, 2)
            for y, x in img_points:
                points.append((x, img_height - y))
        x, y = zip(*points)
        ax.hist2d(
            x=x,
            y=y,
            bins=(img_width // 4, img_height // 4),
            range=[[0, img_width], [0, img_height]],
        )
        ax.set_title(label)

    plot_hist(gt_ax, gt_batch, "GT distribution")
    plot_hist(pr_ax, pred_batch, "Prediction distribution")

    plt.savefig("distrib.jpeg")
    plt.close()
    image = cv2.imread("distrib.jpeg", cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
