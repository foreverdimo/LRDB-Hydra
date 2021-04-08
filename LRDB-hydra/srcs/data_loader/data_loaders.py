import random
from random import randint
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from srcs.data_loader.pipal_dataset import PIPAL_Dataset, PIPAL_Paired_Dataset, PIPAL_Test_Dataset
from srcs.utils.util import instantiate

def get_data_loaders(datasets, batch_size, shuffle=True, num_workers=1):
    trainloader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers
    }
    valloader_args = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 1
    }
    (train_dataset, val_dataset) = instantiate(datasets)
    return (DataLoader(train_dataset, **trainloader_args), DataLoader(val_dataset, **valloader_args))

def get_datasets(dst_dir, ref_dir, label_dir=None, train_mode='single'):
    l = list(range(200))
    random.shuffle(l)
    train_list = l[:180]
    val_list = l[180:]
    val_list.sort()

    if label_dir is None:
        test_dataset = PIPAL_Test_Dataset(dst_dir, ref_dir)
        return test_dataset
        
    if train_mode == 'single':
        train_dataset = PIPAL_Dataset(dst_dir, ref_dir, label_dir, train_list)
    elif train_mode == 'paired':
        train_dataset = PIPAL_Paired_Dataset(dst_dir, ref_dir, label_dir, train_list)
    else:
        raise ValueError
    val_dataset = PIPAL_Dataset(dst_dir, ref_dir, label_dir, val_list)
    return (train_dataset, val_dataset)