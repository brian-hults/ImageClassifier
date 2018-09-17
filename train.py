# Imports here
import utils
import model_fns
from collections import OrderedDict
import torch
import argparse
import json
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Get user input to modify the model
p = argparse.ArgumentParser()
p.add_argument('data_directory')
p.add_argument('--save_dir', default='/home/workspace/aipnd-project')
p.add_argument('--arch', help='vgg16 or vgg19', default='vgg16')
p.add_argument('--learning_rate', default=0.001)
p.add_argument('--hidden_units', default=512)
p.add_argument('--epochs', default=1)
p.add_argument('--gpu', default='gpu')
args = p.parse_args()

# Load the data for training, validation, and testing
data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms for datasets
train_transforms = transforms.Compose([transforms.RandomRotation(60),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(), 
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

# Load datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Define dataloaders using datasets and transforms
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

# Load mapping from category label to category name
with open('/home/workspace/aipnd-project/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Load in a pretrained model (default = VGG16)
if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
else:
    model = models.vgg19(pretrained=True)

# Build a new classifier to add on to the pretrained model
for param in model.parameters():
    param.requires_grad = False
    
input_size = 25088
hidden_size = [4096, args.hidden_units]
output_size = 102  # number of flower categories
    
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_size[0])),
    ('relu1', nn.ReLU()),
    ('drop1', nn.Dropout(0.5)),
    ('fc2', nn.Linear(hidden_size[0], hidden_size[1])),
    ('relu2', nn.ReLU()),
    ('drop2', nn.Dropout(0.5)),
    ('fc3', nn.Linear(hidden_size[1], output_size)),
    ('output', nn.LogSoftmax(dim=1))
]))
model.classifier = classifier

# Setup error criterion and optimizer function
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate) # Only train the classifier parameters!

# Sets which device training will run on depending on user input
if args.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'

# Call function to train and validate the model
model_fns.training(model, args.epochs, device, trainloader, validloader, criterion, optimizer)

# Test the trained model on the test data
model_fns.testing(model, args.epochs, device, testloader, criterion)

# Save the trained model
utils.save_checkpoint(model, optimizer, args.save_dir, args.epochs, args.learning_rate, args.hidden_units, train_data, args.arch)