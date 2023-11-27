import os
import fnmatch
import torch
from torch.utils.data import Dataset
from PIL import Image


class PetNoseDataset(Dataset):

    def __init__(self, root, images_dir='images', train=True, transform=None):
        self.root = root
        self.transform = transform

        self.train = train
        labels_name = 'train_noses.3.txt' if train else 'test_noses.txt'

        self.noses = dict()
        with open(os.path.join(self.root, labels_name)) as label_file:
            for line in label_file.readlines():
                line = line.strip()
                if line != '':
                    image, coords = line.split(',', 1)
                    assert coords.startswith('"(')
                    assert coords.endswith(')"')
                    coords = coords[2:-2].split(',')
                    coords = (float(coords[0]), float(coords[1]))
                    self.noses[image] = coords

        self.image_files = []
        self.images_dir = os.path.join(self.root, images_dir)
        for subdir in os.listdir(self.images_dir):
            dir = os.path.join(self.images_dir, subdir)
            if os.path.isdir(dir):
                for image in os.listdir(dir):
                    if fnmatch.fnmatch(image, '*.jpg') and image in self.noses:
                        self.image_files += [(subdir, image)]

        self.max = len(self.image_files)


    def __len__(self):
        return self.max

    def __getitem__(self, idx):
        dir, name = self.image_files[idx]
        nose_pos = self.noses[name]
        path = os.path.join(self.images_dir, dir, name)
        image = Image.open(path).convert('RGB')

        # normalize nose positions
        nose_pos_x = nose_pos[0] / image.size[0]
        nose_pos_y = nose_pos[1] / image.size[1]

        if self.transform:
            image = self.transform(image)
        return [image, torch.tensor(data=(nose_pos_x, nose_pos_y), dtype=torch.float32)]

    def __iter__(self):
        self.num = 0
        return self

    def __next__(self):
        if self.num >= self.max:
            raise StopIteration
        else:
            self.num += 1
            return self.__getitem__(self.num-1)


if __name__ == '__main__':
    train = PetNoseDataset(root='./oxford-iiit-pet-noses', train=True)
    test = PetNoseDataset(root='./oxford-iiit-pet-noses', train=False)

    print(len(train))
    print(len(test))

    train_count = 0
    test_count = 0
    for _ in train:
        train_count += 1
    for _ in test:
        test_count += 1

    print(train_count)
    print(test_count)


