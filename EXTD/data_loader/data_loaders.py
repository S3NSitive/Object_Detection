import sys
import torch
import scipy.io
import numpy as np

sys.path.append(".")
sys.path.append("..")

from skimage import io
from pathlib import Path
from base import BaseDataLoader
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as patches


import torch
import numpy as np

from skimage import transform


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        print(landmarks[0])
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


def show_landmarks(image, landmarks):
        """Show image with landmarks"""
        fig, ax = plt.subplots(1)
        image = torch.transpose(image, 1, 2)
        image = torch.transpose(image, 2, 3)
        ax.imshow(image[0])
        for i in range(len(landmarks[0])):
            rect = patches.Rectangle((landmarks[0][i, 0], landmarks[0][i, 1]), landmarks[0][i, 2], landmarks[0][i, 3], linewidth=1, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
        plt.pause(1)  # pause a bit so that plots are updated


class WIDERDataset(Dataset):
    """WIDER Face dataset"""
    def __init__(self, base_dir, fname, train=False, download=False, transform=None):
        self.label_dir = base_dir
        self.fname = fname
        self.transform = transform
        self.image_dir = self.label_dir / "images"

        self.wider_mat = scipy.io.loadmat(str(self.label_dir / fname))
        self.event_list = self.wider_mat.get("event_list")
        self.file_list = self.wider_mat.get("file_list")
        self.face_bbx_list = self.wider_mat.get("face_bbx_list")

        idx = 0
        self.data_list = []
        for event_idx, event in enumerate(self.event_list):
            for im_idx, im in enumerate(self.file_list[event_idx][0]):
                im_name = im[0][0]
                face_bbx = self.face_bbx_list[event_idx][0][im_idx][0]

                bboxes = []
                for i in range(face_bbx.shape[0]):
                    xmin = int(face_bbx[i][0])
                    ymin = int(face_bbx[i][1])
                    xmax = int(face_bbx[i][2])
                    ymax = int(face_bbx[i][3])
                    bboxes.append((xmin, ymin, xmax, ymax))

                self.data_list.append((event[0][0], im_name, bboxes))
                idx += 1

    def __getitem__(self, index):
        image_name = str(self.image_dir / self.data_list[index][0] / (self.data_list[index][1] + ".jpg"))
        image = io.imread(image_name)
        landmarks = np.array(self.data_list[index][2]).astype("float")
        sample = {"image": image, "landmarks": landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data_list)

    def collate_fn(self, batch):
        images = list()
        boxes = list()

        for sample in batch:
            images.append(sample['image'])
            boxes.append(sample['landmarks'])

        images = torch.stack(images, dim=0)
        return {'image': images, 'landmarks': boxes}


class WIDERDataLoader(BaseDataLoader):
    def __init__(self, base_dir, fname, batch_size, shuffle=False, validation_split=0.0, num_workers=1, training=True):
        """Build a wider parser

        Args:
            data_dir (str):
            fname (str):
        """
        trsfm = transforms.Compose([
            Rescale(640),
            ToTensor()
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.base_dir = Path(base_dir)
        self.dataset = WIDERDataset(self.base_dir, fname, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                         collate_fn=self.dataset.collate_fn)


def main():
    # dataset = WIDERDataset("./data/WIDER_FACE", "wider_face_train.mat")
    # print(dataset[0])
    dataloader = WIDERDataLoader("./data/WIDER_FACE", "wider_face_train.mat", 1)
    for idx, train_iter in enumerate(dataloader):
        show_landmarks(**train_iter)


if __name__ == "__main__":
    main()
