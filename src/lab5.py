import torch
import torch.nn as nn
import torchvision
import argparse

# Contains all the code shared by the test and train modules

def create_model():
    model = torchvision.models.resnet18(pretrained=False)
    # model.fc = nn.Linear(model.fc.in_features, 2)
    model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 2)
    )
    return model

def create_arguments():
    args = argparse.ArgumentParser()
    args.add_argument('model', type=str, help='path to weights file')
    args.add_argument('-device', type=str, default=None, help='device to use for testing (leave blank to autodetect)')
    args.add_argument('-dataset', type=str, default='./oxford-iiit-pet-noses', help='dataset directory')
    return args

def get_device(opts):
    if opts.device is not None:
        return torch.device(opts.device)
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
