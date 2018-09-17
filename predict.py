# Imports here
import utils
import model_fns
import argparse
import json
from collections import OrderedDict
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

p = argparse.ArgumentParser()
p.add_argument('image_directory')
p.add_argument('checkpoint_directory')
p.add_argument('--top_k', default=5)
p.add_argument('--category_names_dir', default='/home/workspace/aipnd-project/cat_to_name.json')
p.add_argument('--gpu', default='gpu')
args = p.parse_args()

# Sets which device training will run on depending on user input
if args.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'

# Preprocess image
np_img = utils.process_image(args.image_directory)
img_tensor = torch.from_numpy(np_img)
img_tensor = torch.unsqueeze(img_tensor, 0)
img_tensor = img_tensor.float()
img_tensor = img_tensor.to(device)

# Load the model with saved features and classifier
model = utils.load_checkpoint(args.checkpoint_directory)

with torch.no_grad():
    model.eval()
    model.to(device)
    output = model.forward(img_tensor)

probs = torch.topk(torch.exp(output), args.top_k)

# Covert CUDA Tensors back to CPU Tensors (if applicable) and then to Numpy Arrays
if device == 'cuda':
    probabilities = probs[0][0].cpu().numpy()
    classes = probs[1][0].cpu().numpy()
else:
    probabilities = probs[0][0].numpy()
    classes = probs[1][0].numpy()

# Inverse the model.class_to_idx dictionary so we can look up which category values match our predictions
idx_to_class = {v: k for k, v in model.class_to_idx.items()}

# Initialize the result classes list and add predicted categories to the list based on our index values
result_classes = []
for i in range(len(classes)):
    result_classes.append(idx_to_class.get(classes[i]))
    
# Extract the json categories file from the given directory
with open(args.category_names_dir, 'r') as f:
    cat_to_name = json.load(f)
    
# Initialize the labels list and add the labels of predicted categories
labels = []
for i in range(len(result_classes)):
    labels.append(cat_to_name.get(result_classes[i]))

# Print results
print('Categories: ', labels)
print('Probabilities: ', probabilities)