import os
import fnmatch
import torch
from torch.utils.data import Dataset
from PIL import Image


class PetNoseDataset(Dataset):

    def __init__(self, root, labels_name='train_noses.3.txt', images_dir='images', train=True, transform=None):
        self.root = root
        self.transform = transform

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
                    if fnmatch.fnmatch(image, '*.jpg') and image in self.noses: # some images are not in labels.txt, skip those
                        self.image_files += [(subdir, image)]

        self.train = train
        if train:
            self.start = 0
            self.end = 5000
        else:
            self.start = 5000
            self.end = len(self.image_files)


    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        dir, name = self.image_files[idx]
        nose_pos = self.noses[name]
        path = os.path.join(self.images_dir, dir, name)
        image = Image.open(path).convert('RGB')

        # normalize nose positions so we can resize the image later and the nose position still has meaning
        nose_pos_x = nose_pos[0] / image.size[0]
        nose_pos_y = nose_pos[1] / image.size[1]

        if self.transform:
            image = self.transform(image)
        return [image, torch.tensor(data=(nose_pos_x, nose_pos_y), dtype=torch.float32)]

    def __iter__(self):
        self.num = self.start
        return self

    def __next__(self):
        if self.num >= self.end:
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


