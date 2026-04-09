"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import random

import torch
import torch.utils.data
from torch.utils.data._utils.collate import default_collate

from cut.data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "cut.data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


def _center_patch_batch(batch, opt):
    collated = default_collate(batch)
    if 'center_patch_batch' not in opt.preprocess:
        return collated

    image_keys = [key for key in ('A', 'B') if key in collated and isinstance(collated[key], torch.Tensor) and collated[key].dim() == 4]
    if not image_keys:
        return collated

    crop_size = opt.crop_size
    reference = collated[image_keys[0]]
    height = reference.shape[2]
    width = reference.shape[3]
    if height <= crop_size and width <= crop_size:
        return collated

    center_x = max(0, (width - crop_size) // 2)
    center_y = max(0, (height - crop_size) // 2)
    max_offset = int(max(0, opt.center_patch_offset))
    x_min_shift = -center_x
    x_max_shift = (width - crop_size) - center_x
    y_min_shift = -center_y
    y_max_shift = (height - crop_size) - center_y

    dx = random.randint(max(-max_offset, x_min_shift), min(max_offset, x_max_shift))
    dy = random.randint(max(-max_offset, y_min_shift), min(max_offset, y_max_shift))

    x0 = center_x + dx
    y0 = center_y + dy
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    for key in image_keys:
        collated[key] = collated[key][:, :, y0:y1, x0:x1]

    return collated


class CenterPatchCollateFn:
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, batch):
        return _center_patch_batch(batch, self.opt)


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
            drop_last=True if opt.isTrain else False,
            collate_fn=CenterPatchCollateFn(opt),
        )

    def set_epoch(self, epoch):
        self.dataset.current_epoch = epoch

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
