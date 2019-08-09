import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from pathlib import Path
from torchvision import transforms, utils
from torch.utils.data import Dataset

from FAN.base import BaseDataLoader
from FAN.data_loader.transforms import Rescale, ToTensor


def show_landmarks(image, output_landmarks, target_landmarks):
    """Show image with landmarks"""
    # image = image.permute((1, 2, 0))
    # target_landmarks = target_landmarks
    batch_size = len(image)
    im_size = image.size(2)
    grid_border_size = 2

    grid = utils.make_grid(image)
    plt.figure()
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    # print(output_landmarks[0, :, 0][0])
    # print(target_landmarks[0, :, 0])
    for i in range(batch_size):
        plt.scatter(target_landmarks[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    target_landmarks[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')
        plt.scatter(output_landmarks[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    output_landmarks[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='b')

    # plt.figure()
    # plt.imshow(image)
    # plt.scatter(target_landmarks[:, 0], target_landmarks[:, 1], s=10, marker='.', c='r')
    # plt.scatter(output_landmarks[:, 0], output_landmarks[:, 1], s=10, marker='.', c='b')
    plt.axis('off')
    plt.ioff()
    plt.show()
    # plt.pause(1)  # pause a bit so that plots are updated


class Alignment300W_Dataset(Dataset):
    """WIDER Face dataset"""
    def __init__(self, base_dir, train=False, download=False, transform=None):
        self.img_dir = base_dir / "png"
        self.pts_dir = base_dir / "pts"
        self.transform = transform

        self.imgs = list(self.img_dir.glob("**/*.png"))
        self.ptss = list(self.pts_dir.glob("**/*.pts"))

    def __getitem__(self, index):
        # cv2.cvtColor(cv2.imread(str(self.imgs[index])), cv2.COLOR_BGR2RGB)
        image = io.imread(str(self.imgs[index]))

        landmarks = []
        with open(str(self.ptss[index]), encoding="utf-8") as f:
            point = f.read().splitlines()
            for i in point[3:-1]:
                landmarks.append(list(map(float, i.split(" "))))

        sample = {"image": image, "landmarks": np.array(landmarks)}

        if self.transform:
            sample = self.transform(sample)


        return sample

    def __len__(self):
        return len(self.imgs)


class Alignment300W_DataLoader(BaseDataLoader):
    def __init__(self, base_dir, batch_size=1, shuffle=False, validation_split=0.0, num_workers=1, training=True):
        """Build a wider parser

        Args:
            data_dir (str):
            fname (str):
        """
        trsfm = transforms.Compose([
            Rescale((256, 256)),
            ToTensor()
        ])
        self.base_dir = Path(base_dir)
        self.dataset = Alignment300W_Dataset(self.base_dir, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


def main():
    dataloader = Alignment300W_DataLoader(base_dir="./data/")
    for idx, train_iter in enumerate(dataloader):
        print(idx+1, train_iter["image"].size(), train_iter["landmarks"].size())
        show_landmarks(**train_iter)


if __name__ == "__main__":
    main()
