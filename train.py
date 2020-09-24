import torch, fire
from torch.utils.data import DataLoader, random_split
from data import LargeImageDataset, FolderDatasetDownsample
from model import GAN
import torchvision
import os
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import glob

def remove_files_in_path(path, ext=""):
    files = glob.glob(os.path.join(path, "*"+ext))
    print(files)
    for f in files:
        os.remove(f)


def train(
        root_path,
        batch_size,
        epochs=1,
        lr=.001,
        max_filters=256,
        min_filters=64,
        upsample_layers=3,
        noise_dim=64,
        blocks=8,
        device_ids=[0], # pass in list of device ids you want to use, if multiple will use DataParallel
        image_size=256,
        batch_shuffle=True,
        num_workers=0,
        wass_target=1,
        mse_weight=10,
        ttur=4,
        models_dir='./models/',
        results_dir='./results/',
        log_dir='./log/',
        model_name_prefix='test',
        save_every=20000,
        print_every=500,
        log_every=1000,
        conv_type='scaled',
        grad_mean=True,
        upsample_type='nearest',
        resume=True,
        checkpoint=-1, #only used if resume is true, load from which last save? default is latest
        straight_through_round=True,
        reset=False,
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
    sample_indices = torch.randint(0, len(dataset), (batch_size,))
    samples_list = [dataset[i] for i in sample_indices]
    lores_list, hires_list = zip(*samples_list)
    lores_samples = torch.stack(lores_list)
    hires_samples = torch.stack(hires_list)
    samples = lores_samples, hires_samples

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=batch_shuffle, pin_memory=False)
    model_name = f'{model_name_prefix}.{datasetname}.{batch_size}.{lr}.{max_filters}.{min_filters}.{upsample_layers}.{blocks}.{noise_dim}.{image_size}.{wass_target}.{ttur}.{mse_weight}'

    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    if not os.path.exists(models_dir + model_name):
        os.mkdir(models_dir + model_name)

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_dir = Path(log_dir) / model_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if reset:
        #erase models, erase logs
        print('resetting', models_dir+model_name, log_dir)
        remove_files_in_path(models_dir + model_name, 'pt')
        remove_files_in_path(log_dir, '0')


    writer = SummaryWriter(log_dir=log_dir)
    model = GAN(max_filters=max_filters, min_filters=min_filters,
            upsample_layers=upsample_layers, noise_dim=noise_dim,
            blocks=blocks, device_ids=device_ids, models_dir=models_dir,
            results_dir=results_dir, log_writer=writer, model_name=model_name,
            save_every=save_every, print_every=print_every,
            log_every=log_every, conv_type=conv_type, grad_mean=grad_mean,
            upsample_type=upsample_type,
            straight_through_round=straight_through_round,
            samples=samples)
    if resume:
        model.load(checkpoint)

    n_iter = model.train(dataloader, epochs, lr, wass_target, mse_weight, ttur)
    model.save(n_iter)
    writer.close()


if __name__ == '__main__':
  fire.Fire(train)
