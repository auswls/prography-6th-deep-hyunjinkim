import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision
from matplotlib.pyplot import imshow
import torchvision.utils as utils
import torchvision.datasets as dsets
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage
import glob
import matplotlib.image as mpimg

from model import VGG16

parser = argparse.ArgumentParser(description='Prography-6th-assignment-HyunjinKim')
"""
parser.add_argument('--dataroot', default="/input/" ,help='path to dataset')
parser.add_argument('--evalf', default="/eval/" ,help='path to evaluate sample')
parser.add_argument('--outf', default='models',
        help='folder to output images and model checkpoints')
parser.add_argument('--ckpf', default='',
        help="path to model checkpoint file (to continue training)")
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
        help='input batch size for training (default: 4)')
parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
        help='input batch size for testing (default: 4)')
parser.add_argument('--epochs', type=int, default=3, metavar='N',
        help='number of epochs to train (default: 3)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
        help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
        help='SGD momentum (default: 0.5)')
"""
parser.add_argument('--no-cuda', action='store_true', default=False,
        help='disables CUDA training')
"""
parser.add_argument('--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
        help='how many batches to wait before logging training status')
parser.add_argument('--train', action='store_true',
        help='training a VGG16 modified model on MNIST dataset')
parser.add_argument('--evaluate', action='store_true',
        help='evaluate a [pre]trained model')
"""


args = parser.parse_args()


use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


# transform to rgb
rgb_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
             ])


def inference(model, sample_path) : 
  sample_dataset = glob.glob(sample_path)

  for image in sample_dataset : 
    img = Image.open(image)
    img = rgb_transform(img) # size : [3,224,224]
    img = img.unsqueeze(0) # size : [1,3,224,224]
    #print("size of extended image : ", img.size())
    input_image = img.to(device)
    
    output = model(input_image)
    prediction = output.max(dim=1)[1].item()
    img2 = mpimg.imread(image)
    plt.imshow(img2)
    plt.title("prediction : "+str(prediction))
    plt.show()

    print("Prediction result : ", prediction)


model = VGG16().to(device)
model.load_state_dict(torch.load("/content/drive/My Drive/prography/model/mnist_convnet_model_epoch_2.pth"))


sample_image_path = "/content/drive/My Drive/prography/sample/*jpg" # put sample image to sample file as jpg extension

inference(model, sample_image_path)
