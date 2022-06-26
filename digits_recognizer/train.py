import argparse
from datetime import datetime
from nbformat import write
import numpy as np
import os
import platform

from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torchvision
from tqdm import tqdm
import yaml

from dataloader import MNISTDataset
from digits_recognizer import build_model
from preprocessor import build_preprocessor
from utils import is_scalar, plot_conf_matrix


def get_available_device(verbose=True):
    exec_device = torch.device('cpu')
    if torch.has_mps:
        exec_device = torch.device('mps')
    if torch.has_cuda:
        exec_device = torch.device('cuda')
    if verbose:
        print(f'Platform: {platform.system()}')
        print(f'Release: {platform.release()}')
        print(f'MPS available: {torch.has_mps}')
        print(f'CUDA available: {torch.has_cuda}')
        print(f'Selected device: {exec_device}')
    return exec_device


def train(
    model,
    epochs,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_function,
    train_metrics=None,
    val_metrics=None,
    val_period=0,
    autosave_period=0,
    logs_root="logs",
    checkpoints_root="checkpoints",
    device=None,
):
    timestamp = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
    logs_dir = os.path.join(logs_root, timestamp)
    models_dir = os.path.join(checkpoints_root, timestamp)
    os.makedirs(logs_dir)
    os.makedirs(models_dir)
    writer = SummaryWriter(logs_dir)

    exec_device = get_available_device()
    if device is not None:
        exec_device = torch.device(device)
        print(f'Execution device is overloaded as {exec_device}')
    
    model.to(exec_device)
    model.train()
    total_steps = epochs * len(train_dataloader)
    step_idx = 0
    val_iter = iter(val_dataloader)

    def evaluate(predicted, target, metrics, hint):
        if metrics is None:
            return
        for metric_name, metric_callback in metrics.items():
            value = metric_callback(
                predicted=predicted.to('cpu').detach().numpy(),
                target=target.to('cpu').detach().numpy(),
            )
            if is_scalar(value):
                writer.add_scalar(f'Metrics/{hint}:{metric_name}', value, step_idx)
            else:
                images = [torch.Tensor(value)]
                grid = torchvision.utils.make_grid(images)
                writer.add_image(f'Matrics/{hint}:{metric_name}', grid, step_idx)

    with tqdm(total=total_steps) as pbar:
        for epoch_idx in range(epochs):
            for train_batch in train_dataloader:
                inputs, labels = train_batch
                inputs, labels = inputs.to(exec_device), labels.to(exec_device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                evaluate(predicted=outputs, target=labels, metrics=train_metrics, hint="Train")

                if val_period > 0 and (step_idx % val_period == 0):
                    inputs, labels = next(val_iter)
                    inputs = inputs.to(exec_device)
                    model.eval()
                    outputs = model(inputs)
                    model.train()
                    evaluate(predicted=outputs, target=labels, metrics=val_metrics, hint="Val")

                pbar.set_description(f"Iteration {step_idx}/{total_steps}\tLoss {loss.detach().cpu().numpy()}")
                pbar.update(1)
                step_idx += 1

            if autosave_period > 0 and (epoch_idx % autosave_period == 0) and epoch_idx:
                checkpoint_path = os.path.join(models_dir, f"state_dict_epoch_{epoch_idx}")
                torch.save(model.state_dict(), checkpoint_path)

    print('Training completed')


def build_optimizer(model, optim_config):
    optimizer_pars = {'params': model.parameters()}
    for par, value in optim_config['pars'].items():
        optimizer_pars[par] = value
    return getattr(torch.optim, optim_config['type'])(**optimizer_pars)


def accuracy_evaluator(predicted, target):
    predicted = np.argmax(predicted, axis=-1).flatten()
    target = target.flatten()
    return accuracy_score(y_true=target, y_pred=predicted)


def cm_evaluator(predicted, target):
    predicted = np.argmax(predicted, axis=-1).flatten()
    target = target.flatten()
    cm_image_hwc = plot_conf_matrix(
        confusion_matrix(y_true=target, y_pred=predicted)
    )
    return np.transpose(cm_image_hwc, axes=[2, 0, 1])


def parse_args():
    parser = argparse.ArgumentParser(description='Train digits recognizer')
    parser.add_argument('--config', type=str, default='train.yaml')
    return parser.parse_args()


def run_training(args):
    config = None
    with open(args.config, 'rt') as f:
        config = yaml.safe_load(f.read())
    
    model = build_model(config['model']['type'])
    preprocessor = build_preprocessor(config['model']['preprocessing'])
    img_transform = torchvision.transforms.Compose([
        preprocessor,
    ])

    dataset = MNISTDataset(config['dataset']['path'], transform=img_transform)
    val_target_size = int(config['dataset']['val_share'] * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_target_size, val_target_size])
    train_dataloader = DataLoader(train_dataset, batch_size=config['dataset']['train_batch'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['dataset']['val_batch'], shuffle=True)

    optimizer = build_optimizer(model, config['optimizer'])
    loss_function = torch.nn.CrossEntropyLoss()
    train_metrics = {
        'accuracy': accuracy_evaluator,
    }
    val_metrics = {
        'accuracy': accuracy_evaluator,
        'conf_matrix': cm_evaluator,
    }

    train(
        model,
        config['scheduler']['epochs'],
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_function,
        train_metrics,
        val_metrics,
        config['evaluation']['period'],
        config['autosave']['period'],
        logs_root="logs",
        checkpoints_root="checkpoints",
        device=None,
    )


if __name__ == "__main__":
    run_training(parse_args())
