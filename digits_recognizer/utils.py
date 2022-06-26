from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import tempfile


def is_scalar(value):
    try:
        float(value)
        return True
    except Exception:
        return False


def plot_conf_matrix(matrix):
    plt.figure(figsize = (6,6))
    sns.heatmap(matrix, annot=True)
    temp_file = os.path.join(tempfile.gettempdir(), "confmat.jpg")
    plt.savefig(temp_file)
    return np.asarray(Image.open(temp_file))


def plot_samples(samples, cols=4):
    sample_size = len(samples)
    rows = sample_size // cols
    if sample_size % cols:
        rows += 1
    fig, m_axs = plt.subplots(rows, cols, figsize = (16, 16))
    for (img, label), c_ax in zip(samples, m_axs.flatten()):
        c_ax.imshow(img, cmap = 'gray')
        c_ax.set_title(f'{label}')
        c_ax.axis('off')
