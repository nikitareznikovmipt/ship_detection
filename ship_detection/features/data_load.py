import os
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import xml.etree.ElementTree as ET

# albumentation for augmentations 

class ShipDataset_1(Dataset):
    """
    Class for load dataset_1
    """
    def __init__(self, image_dir, annotation_dir, transforms=None):
        """
        :param image_dir: path to images
        :param annotation_dir: path to xml annotations
        """
        self.image_dir = image_dir
        self.bndboxes = annotation_dir

        xml_tree = ET.parse(os.path.join(self.bndboxes, f'boat{item}.xml'))
        root = xml_tree.getroot()
        self.bndboxes = []

        for neighbor in root.iter('bndbox'):
            xmin = int(neighbor.find('xmin').text)
            ymin = int(neighbor.find('ymin').text)
            xmax = int(neighbor.find('xmax').text)
            ymax = int(neighbor.find('ymax').text)
            self.bndboxes.append([xmin, ymin, xmax, ymax])

        # Одинаковая сортировка
        self.files = []
        self.targets = [] 
        self.transforms = transforms
    def __len__(self):
        """
        :return: number of element in dataset
        """
        return len(os.listdir(self.bndboxes))

    def __getitem__(self, idx):
        """
        :param item: index of image
        :return: image and bndboxes in list format
        """
        img: torch.Tensor = read_image(os.path.join(self.image_dir, f'boat{item}.png'))

        if self.transforms is not None:
            # Random crop 
            # Resize 
            img = self.transforms(img)
        return img, bndboxes



class ShipDataset_2(Dataset):
    """
    Class for load dataset_2
    """
    def __init__(self, image_dir, annotation_file):
        """
        :param image_dir: path to images
        :param annotation_file: path to csv annotations
        """
        self.image_dir = image_dir
        self.labels = pd.read_csv(annotation_file).fillna(0)

    def __len__(self):
        """
        :return: number of element in dataset
        """
        return len(self.labels)

    def __getitem__(self, item):
        """
        :param item: index of image
        :return: image and encoded pixels
        """
        item = 1
        img_name = self.labels.iloc[item, 0]
        label = self.labels.iloc[item, 1]
        img_path = os.path.join(self.image_dir, img_name)
        img = read_image(img_path)
        # RLE - run lengthon, polygon

        return img, label


if __name__ == '__main__':
    # Проверка корректности работы первого датасета
    dataset_1 = ShipDataset_1('/Users/nikitareznikov/Desktop/DS/ship_detection/data/raw/dataset_1/images',
                              '/Users/nikitareznikov/Desktop/DS/ship_detection/data/raw/dataset_1/annotations')
    # print(len(dataset_1))
    # print(dataset_1[0])
    # Проверка корректности работы второго датасета
    dataset_2 = ShipDataset_2('/Users/nikitareznikov/Desktop/DS/ship_detection/data/raw/dataset_2/train',
                              '/Users/nikitareznikov/Desktop/DS/ship_detection/data/raw/dataset_2'
                              '/train_ship_segmentations.csv')
    # print(len(dataset_2))
    # print(dataset_2[2])

    dataloader_1 = DataLoader(
        dataset=dataset_1, batch_size=64)
    dataloader_2 = DataLoader(
        dataset=dataset_2, batch_size=64
        )

    for img, bndboxes in dataloader_1:
        print(img.shape, bndboxes)
        break

    # (346, 246),
    # (256, 346),


