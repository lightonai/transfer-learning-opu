import os
from torch.utils.data.dataset import Dataset
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