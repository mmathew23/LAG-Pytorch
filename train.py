import torch, fire
from torch.utils.data import DataLoader
from data import LargeImageDataset, FolderDatasetDownsample
from model import GAN
import torchvision
import os
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def train(
        root_path,
        batch_size,
        epochs=1,
        lr=.0001,
        max_filters=256,
        min_filters=64,
        upsample_layers=3,
        noise_dim=64,
        blocks=8,
        device_id=0,
        image_size=256,
        batch_shuffle=True,
        num_workers=4,
        wass_target=1,
        mse_weight=10,
        ttur=4,
        models_dir='./models/',
        results_dir='./results/',
        log_dir='./log/',
        model_name_prefix='test',
        save_every=10000
        ):
    #device = "cpu"
    #if torch.cuda.is_available():
    #    device = f'cuda:{device_id}'
    #create dataset/loader
    datasetname = ''
    if os.path.isdir(root_path):
        dataset = FolderDatasetDownsample(root_path, downsample=2**upsample_layers, size=image_size)
        datasetname='folder'
    else:
        dataset = LargeImageDataset(root_path, downsample=2**upsample_layers, size=image_size)
        datasetname='largeimage'

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=batch_shuffle, pin_memory=False)
    model_name = f'{model_name_prefix}.{datasetname}.{epochs}.{batch_size}.{lr}.{max_filters}.{min_filters}.{upsample_layers}.{blocks}.{noise_dim}.{image_size}.{wass_target}.{ttur}.{mse_weight}'

    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    if not os.path.exists(models_dir + model_name):
        os.mkdir(models_dir + model_name)

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_dir = Path(log_dir) / model_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    model = GAN(max_filters=max_filters, min_filters=min_filters,
            upsample_layers=upsample_layers, noise_dim=noise_dim,
            blocks=blocks, device_id=device_id, models_dir=models_dir,
            results_dir=results_dir, log_writer=writer, model_name=model_name,
            save_every=save_every)

    model.train(dataloader, epochs, lr, wass_target, mse_weight, ttur)
    writer.close()


if __name__ == '__main__':
  fire.Fire(train)
