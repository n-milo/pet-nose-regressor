import math
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import cv2
import numpy as np

import lab5
from dataset import PetNoseDataset
from tqdm import tqdm

import matplotlib.pyplot as plt


def test_model(model, device, should_display, dataset_dir, output_file):
    print('Testing model...')

    test_dataset = PetNoseDataset(root=dataset_dir, train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    resizer = transforms.Resize((224, 224), antialias=True)

    dists_norm = []
    dists_pixel = []

    for images, labels in tqdm(test_loader):
        resized = resizer(images[0].cpu()).unsqueeze(0).to(device)
        labels = labels.to(device)
        outputs = model(resized)

        output_x_norm = outputs[0][0].item()
        output_y_norm = outputs[0][1].item()
        label_x_norm = labels[0][0].item()
        label_y_norm = labels[0][1].item()

        w = images[0].shape[2]
        h = images[0].shape[1]
        output_x = int(output_x_norm * w)
        output_y = int(output_y_norm * h)
        label_x = int(label_x_norm * w)
        label_y = int(label_y_norm * h)

        dist_pixels = math.sqrt((output_x - label_x) ** 2 + (output_y - label_y) ** 2)
        dist_norm = math.sqrt((output_x_norm - label_x_norm) ** 2 + (output_y_norm - label_y_norm) ** 2)

        dists_pixel.append(dist_pixels)
        dists_norm.append(dist_norm)

        if should_display:
            display = images[0].cpu().numpy().transpose(1, 2, 0).copy()
            cv2.circle(display, (output_x, output_y), 3, (0, 255, 0), -1)
            cv2.circle(display, (output_x, output_y), 15, (0, 255, 0), 1)
            cv2.circle(display, (label_x, label_y), 3, (0, 0, 255), -1)
            cv2.circle(display, (label_x, label_y), 15, (0, 0, 255), 1)
            cv2.imshow('Image', display)
            key = cv2.waitKey(0)
            if key == ord('c'):
                should_display = False
            elif key == ord('q'):
                break
            else:
                continue

    print_results(sys.stdout, dists_pixel, dists_norm)
    if output_file is not None:
        print_results(output_file, dists_pixel, dists_norm)


def print_results(file, dists_pixel, dists_norm):
    print('PET NOSE REGGRESSOR RESULTS', file=file)
    print('\tpixels\tnormalized', file=file)
    print(f'min:\t{np.min(dists_pixel):.1f}\t{np.min(dists_norm):.4f}', file=file)
    print(f'max:\t{np.max(dists_pixel):.1f}\t{np.max(dists_norm):.4f}', file=file)
    print(f'mean:\t{np.mean(dists_pixel):.1f}\t{np.mean(dists_norm):.4f}', file=file)
    print(f'stdev:\t{np.std(dists_pixel):.1f}\t{np.std(dists_norm):.4f}', file=file)
    print('(normalized values: 0.0 = top left, 1.0 = bottom right)', file=file)


if __name__ == '__main__':
    args = lab5.create_arguments()
    args.add_argument('-no_display', action='store_true', help='do not display sample images during testing')
    args.add_argument('-output', type=str, default=None, help='save test output to a file')
    opts = args.parse_args()

    should_display = not opts.no_display
    device = lab5.get_device(opts)

    model = lab5.create_model()
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load(opts.model))

    print('PET NOSE TESTER')
    print('===============')
    print('C - finish testing without display')
    print('Q - quit immediately')
    print('any other key - display next image')
    print('blue: predicted, red: actual')

    output_file = None
    if opts.output is not None:
        output_file = open(opts.output, 'w')

    test_model(model, device,
               should_display=should_display,
               dataset_dir=opts.dataset,
               output_file=output_file)


