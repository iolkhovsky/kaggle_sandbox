import argparse
import datetime
import numpy as np
import os
import platform
import yaml
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchsummary import summary
import traceback

from dataset.dataloader import KeyPointsDataset
import dataset.transforms as xforms
from dataset.visualization import visualize_training
from model.keypoints_regressor import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints regressor')
    parser.add_argument('--config', type=str, default='configs/train.yaml')
    return parser.parse_args()


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


def save_model(model, models_dir, hint, save_onnx=False):
    checkpoint_path = os.path.join(models_dir, f"state_dict_{hint}")
    torch.save(model.state_dict(), checkpoint_path)
    if save_onnx:
        init_device = next(model.parameters()).device
        model.eval()
        model = model.to("cpu")
        onnx_path = os.path.join(models_dir, f"model_{hint}.onnx")
        input_names = ["input0"]
        output_names = ["output0"]
        img_height = model.preprocessor._height
        img_width = model.preprocessor._width
        dummy_input = torch.randn(1, 3, img_height, img_width).to("cpu").float()

        torch.onnx.export(
            model, dummy_input, onnx_path, verbose=True, input_names=input_names,
            output_names=output_names,
        )

        model.to(init_device)
        model.train()
    print(f"Model has been saved to {models_dir}")


def build_datasets(data_config):
    transforms = []
    for transform in data_config['transforms']:
        op = getattr(xforms, transform['transform'])
        args = {}
        if 'attrs' in transform:
            args = transform['attrs']
        transforms.append(op(**args))
    dataset_transform = xforms.CompositeTransform(transforms)

    full_dataset = KeyPointsDataset(data_config['path'], transform=dataset_transform)
    full_size = len(full_dataset)
    val_size = int(full_size * data_config['val']['share'])
    test_size = int(full_size * data_config['test']['share'])
    train_size = full_size - val_size - test_size
    assert train_size > 0, f"train_size = {train_size}"
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size]
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=data_config['train']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=data_config['val']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=data_config['test']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    return train_loader, val_loader, test_loader


def build_optimizer(model, optim_config):
    optimizer_pars = {'params': model.parameters()}
    for par, value in optim_config['pars'].items():
        optimizer_pars[par] = value
    return getattr(torch.optim, optim_config['type'])(**optimizer_pars)


def build_scheduler(optimizer, scheduler_config):
    scheduler_pars = {'optimizer': optimizer}
    for par, value in scheduler_config['pars'].items():
        scheduler_pars[par] = value
    return getattr(torch.optim.lr_scheduler, scheduler_config['type'])(**scheduler_pars)


def run_training(config):
    train_loader, val_loader, test_loader = build_datasets(config['dataset'])
    model = build_model(config['model'])
    criterion = getattr(torch.nn, config['training']['criterion'])()
    optimizer = build_optimizer(model, config['training']['optimizer'])
    scheduler = build_scheduler(optimizer, config['training']['scheduler'])

    timestamp = datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
    logs_dir = os.path.join(config['training']['logs_path'], timestamp)
    models_dir = os.path.join(config['training']['autosave']['path'], timestamp)
    save_onnx = config['training']['autosave']['onnx']['enable']
    os.makedirs(logs_dir)
    os.makedirs(models_dir)
    writer = SummaryWriter(logs_dir)
    exec_device = get_available_device()
    # summary(
    #     model,
    #     input_size=(3, config['model']['target_resolution']['width'],
    #         config['model']['target_resolution']['height'])
    # )
    autosave_period = config['training']['autosave']['period']

    train_metrics = {
        'mse': torch.nn.MSELoss(),
        'l1': torch.nn.L1Loss(),
    }
    train_metrics_interval = config['training']['metrics']['train_steps']
    val_metrics = {
        'mse': torch.nn.MSELoss(),
        'l1': torch.nn.L1Loss(),
    }
    val_metrics_interval = config['training']['metrics']['val_steps']

    model.to(exec_device)
    model.train()

    epochs = config['training']['epochs']
    total_steps = epochs * len(train_loader)
    step_idx = 0
    val_iter = iter(val_loader)

    if scheduler is not None:
        writer.add_scalar(f'LearningRate', scheduler.get_last_lr()[0], step_idx)

    with tqdm(total=total_steps) as pbar:
        for epoch_idx in range(epochs):
            for train_batch in train_loader:
                description = f"Epoch {epoch_idx + 1}/{epochs} "
                description += f"Iteration {step_idx}/{total_steps} "

                try:
                    inputs, targets = train_batch['image'], train_batch['keypoints']
                    inputs, targets = inputs.to(exec_device), targets.to(exec_device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                except Exception:
                    print(f"Error: Got an unhandled exception during iteration {step_idx}")
                    print(traceback.format_exc())

                train_loss = loss.detach().cpu().numpy()
                description += f"Loss {train_loss} "
                writer.add_scalar(f'Loss/Train', train_loss, step_idx)

                try:
                    if train_metrics is not None:
                        if step_idx % train_metrics_interval == 0:
                            model.eval()
                            for metric_name, evaluator in train_metrics.items():
                                value = evaluator(model(inputs), targets)
                                writer.add_scalar(f'Metrics/Train":{metric_name}', value, step_idx)
                            model.train()

                    if val_metrics is not None:
                        if step_idx % val_metrics_interval == 0:
                            model.eval()
                            for metric_name, evaluator in val_metrics.items():
                                predicted = model(inputs)
                                value = evaluator(predicted, targets)
                                writer.add_scalar(f'Metrics/Val":{metric_name}', value, step_idx)
                            model.train()
                        
                        np_images = inputs.detach().cpu().numpy().astype(np.uint8)
                        np_pred = predicted.detach().cpu().numpy()
                        np_gt = targets.detach().cpu().numpy()

                        imgs = visualize_training(
                            image_batch=np_images,
                            pred_batch=np_pred,
                            gt_batch=np_gt,
                            ret_images=True,
                        )

                        images = [torch.permute(torch.from_numpy(x), (2, 0, 1)) for x in imgs]
                        grid = torchvision.utils.make_grid(images)
                        writer.add_image(f'Prediction', grid, step_idx)
                except Exception:
                    print(f"Error: Got an unhandled exception during validation step {step_idx}")
                    print(traceback.format_exc())

                pbar.set_description(description)
                step_idx += 1
                pbar.update(1)
            
            try:
                if autosave_period > 0 and (epoch_idx % autosave_period == 0):
                    save_model(model, models_dir, f"epoch_{epoch_idx}", save_onnx=save_onnx)
                
                if scheduler is not None:
                    writer.add_scalar(f'LearningRate', scheduler.get_last_lr()[0], step_idx)
                    scheduler.step()
            except Exception:
                pass

    save_model(model, models_dir, f"epoch_{epoch_idx}_final", save_onnx=save_onnx)
    print('Training completed')


if __name__ == '__main__':
    args = parse_args()
    assert os.path.isfile(args.config)
    with open(args.config, 'rt') as f:
        config = yaml.safe_load(f.read())
    run_training(config)
