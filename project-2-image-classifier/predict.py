# Usage: python predict.py ./flowers/test/9/image_06413.jpg checkpoint.pth

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

ap = argparse.ArgumentParser(description='Predict.py')
ap.add_argument('input', default='./flowers/test/9/image_06413.jpg', nargs='?', action="store", type = str)
ap.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
ap.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = ap.parse_args()
path_image = pa.input
number_of_outputs = pa.top_k

path = pa.checkpoint
pa = ap.parse_args()

def main():
    model = funcs.load_checkpoint(path)
    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)
    probabilities = funcs.predict(path_image, model, number_of_outputs)
    labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    probability = np.array(probabilities[0][0])
    i = 0
    while i < number_of_outputs:
        print(f"{labels[i]} with a probability of {probability[i]}")
        i += 1
    print("Predicting Completed!")
    
if __name__== "__main__":
    main()