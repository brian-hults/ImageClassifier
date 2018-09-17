# Imports here
import utils
import model_fns
from collections import OrderedDict
from PIL import Image
import torch
import argparse
import json
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Utility functions for image processing
def process_image(image):
    means = [0.485, 0.456, 0.406]
    stdevs = [0.229, 0.224, 0.225]
    size = 256, 256
    
    # Open and resize the image
    img = Image.open(image)
    img.thumbnail(size)
    width, height = img.size
    
    # Center crop the image
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    
    # Conver to a Numpy array and normalize/transpose
    np_img = np.array(img)
    np_img = np_img / 255
    np_img = (np_img - means) / stdevs
    np_img = np_img.transpose(2, 0, 1)
    
    return np_img

def save_checkpoint(model, optimizer, filepath, epochs, learnrate, hidden_units, train_data, arch):
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'arch': arch,
                  'model_classifier':model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'model_state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')
    print('Checkpoint Saved!')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['model_classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model