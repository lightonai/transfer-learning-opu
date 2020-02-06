import os
import glob
from random import shuffle

import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler

from torchvision.datasets.folder import default_loader


def make_dataset(images_path, annotation_path):
    samples = []
    with open(annotation_path, "r") as file:
        for line in file.readlines():
            image, class_idx, species, breed = line.rsplit()

            # I subtract 1 to the class label because pytorch wants classes starting from 0.
            item = (os.path.join(images_path, "{}.jpg".format(image)), int(class_idx)-1)
            samples.append(item)
            
    return samples

class CatsAndDogs(Dataset):

    def __init__(self, dataset_path, mode, loader=default_loader, transform=None, target_transform=None):
        """
        Parameters
        ----------
        dataset_path: str,
            path to the folder containing the frames in .jpg format.
        annotation_path: str,
            path to the folder containing the annotation .txt files.
        set_type: str,
            choose between trainval, test.
        loader: Pytorch loader,
            loader for the images. Defaults to the default loader (PIL).
        transform: torch.transform,
            transformation to apply to the samples.
        target_transform: torch.transform,
            transformation to apply to the targets (i.e. labels).
        """

        self.dataset_path = dataset_path
        self.mode = mode

        self.images_path = os.path.join(dataset_path, "images", "images")
        self.annotation_path = os.path.join(dataset_path, "annotations", "annotations", "{}.txt".format(self.mode))
        

        self.transform = transform
        self.target_transform = target_transform

        samples = make_dataset(self.images_path, self.annotation_path)
        
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.loader = loader
        self.classes = [i for i in range(37)]
    

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample, target = self.samples[index]
        sample = self.loader(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def make_bird_dataset(dataset_path):
    idx_to_image = {}
    train_list, test_list = [], []

    # First grab the images indeces
    with open(os.path.join(dataset_path, "images.txt"), "r") as file:

        for line in file.readlines():
            idx, image_name = line.rsplit()
            idx_to_image[idx] = os.path.join(dataset_path, "images", image_name)

    # Then the class labels
    with open(os.path.join(dataset_path, "image_class_labels.txt"), "r") as file:
        for line in file.readlines():
            image_idx, class_idx = line.rsplit()
            idx_to_image[image_idx] = (idx_to_image[image_idx], int(class_idx) - 1)

    # Then split in train and test
    with open(os.path.join(dataset_path, "train_test_split.txt"), "r") as file:
        for line in file.readlines():
            idx, is_train = line.rsplit()
            if int(is_train) == 1:
                train_list.append(idx_to_image[idx])
            else:
                test_list.append(idx_to_image[idx])

    return train_list, test_list


class CUB_200(Dataset):

    def __init__(self, dataset_path, mode, loader=default_loader, transform=None, target_transform=None):
        """
        Parameters
        ----------
        dataset_path: str,
            path to the folder containing the frames in .jpg format.
        annotation_path: str,
            path to the folder containing the annotation .txt files.
        set_type: str,
            choose between trainval, test.
        loader: Pytorch loader,
            loader for the images. Defaults to the default loader (PIL).
        transform: torch.transform,
            transformation to apply to the samples.
        target_transform: torch.transform,
            transformation to apply to the targets (i.e. labels).
        """

        self.dataset_path = dataset_path
        self.mode = mode

        self.transform = transform
        self.target_transform = target_transform

        samples_train, samples_test = make_bird_dataset(self.dataset_path)

        samples = samples_train if mode == "train" else samples_test

        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.loader = loader
        self.classes = [i for i in range(200)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample, target = self.samples[index]
        sample = self.loader(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def make_animalsdataset(path, test_ratio, class_to_idx):
    train_samples, test_samples = [], []

    for class_folder in os.listdir(path):
        class_images = sorted(glob.glob(os.path.join(path, class_folder, "*.jpeg")))
        n_train_images = int(len(class_images) * (1 - test_ratio / 100))

        train_samples.extend([(image, class_to_idx[class_folder]) for image in class_images[:n_train_images]])
        test_samples.extend([(image, class_to_idx[class_folder]) for image in class_images[n_train_images:]])

    return train_samples, test_samples


class Animals10(Dataset):

    def __init__(self, dataset_path, test_ratio, mode="train", loader=default_loader,
                 transform=None, target_transform=None):
        """
        Parameters
        ----------
        dataset_path: str,
            path to the folder containing the frames in .jpg format.
        annotation_path: str,
            path to the folder containing the annotation .txt files.
        set_type: str,
            choose between trainval, test.
        loader: Pytorch loader,
            loader for the images. Defaults to the default loader (PIL).
        transform: torch.transform,
            transformation to apply to the samples.
        target_transform: torch.transform,
            transformation to apply to the targets (i.e. labels).
        """

        self.dataset_path = dataset_path
        self.test_ratio = test_ratio
        self.mode = mode

        self.transform = transform
        self.target_transform = target_transform
        self.class_idx = self.class_to_idx()

        samples_train, samples_test = make_animalsdataset(self.dataset_path, self.test_ratio, self.class_idx)

        samples = samples_train if self.mode == "train" else samples_test

        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.loader = loader
        self.classes = [i for i in range(10)]

    def __len__(self):
        return len(self.samples)

    def class_to_idx(self):
        class_idx = {class_folder: i for i, class_folder in enumerate(os.listdir(self.dataset_path))}
        return class_idx

    def __getitem__(self, index):

        sample, target = self.samples[index]
        sample = self.loader(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

def get_calibration_loader(loader, cal_batch_size, n_calibration_batches):
    """
    Builds a calibration DataLoader object starting from a Pytorch Dataloader.
    Useful if the calibration images have to be extracted from the training set.

    The calibration loader will inherit the number of workers and transforms of the input loader.
    As a sampler, it uses the SequentialSampler, because I want to load the images in the
    same order. This can be changed according to the application.
    I also drop the last batch to avoid issues with TensorRT.


    Parameters
    ----------
    loader: Pytorch DataLoader,
        dataloader object from which to extract the calibration loader.
    cal_batch_size: int,
        calibrator batch size.
    n_calibration_batches: int,
        number of calibration batches.

    Returns
    -------
    calibrator_loader: Pytorch Dataloader,
        loader containing the calibration images.

    """
    n_calibration_images = n_calibration_batches * cal_batch_size
    print("use {} batches of size {} for a total of {} calibration images".format(n_calibration_batches, cal_batch_size,
                                                                                  n_calibration_images))

    mask = [1 if i < n_calibration_images else 0 for i in range(len(loader.dataset))]
    shuffle(mask)
    mask = torch.Tensor(mask)

    calibrator_loader = DataLoader(loader.dataset, batch_size=cal_batch_size,
                                   sampler=SubsetRandomSampler(np.where(mask)[0]),
                                   num_workers=loader.num_workers, drop_last=True)
    return calibrator_loader
