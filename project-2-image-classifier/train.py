# Usage: python train.py ./flowers

import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import OrderedDict

import json
import argparse
import funcs

ap = argparse.ArgumentParser(description='Train.py')
ap.add_argument('data_dir', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default=0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=10)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=256)

pa = ap.parse_args()
root = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
epochs = pa.epochs

def main():
    trainloader, valid_loader, testloader = funcs.load_data(root)
    model, optimizer, criterion = funcs.network_struct(structure, dropout, hidden_layer1, lr)
    funcs.train_neural_net(model, criterion, optimizer, trainloader, epochs, 20)
    funcs.save_checkpoint(model, path, structure, hidden_layer1, dropout, lr)
    print("Training Completed!")

if __name__== "__main__":
    main()