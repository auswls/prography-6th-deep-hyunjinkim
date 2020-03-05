import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from PIL import Image
import copy
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision

from model import VGG16


parser = argparse.ArgumentParser(description='Prography-6th-assignment-HyunjinKim')
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
parser.add_argument('--no-cuda', action='store_true', default=False,
        help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
        help='how many batches to wait before logging training status')
parser.add_argument('--train', action='store_true',
        help='training a VGG16 modified model on MNIST dataset')
parser.add_argument('--evaluate', action='store_true',
        help='evaluate a [pre]trained model')


args = parser.parse_args()


use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


# transform to rgb
rgb_tranform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
             ])


# MNIST Dataset
test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=rgb_tranform)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)


# model
model = VGG16().to(device)
model.load_state_dict(torch.load("/content/drive/My Drive/prography/model/vgg16_model_epoch_2.pth")) # change the path


def test(args, model, device, test_loader):
    """Testing"""
    #model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    accuracy = 100. * correct / len(test_loader.dataset)
    print('{{"metric": "Eval - Accuracy", "value": {}}}'.format(accuracy))

    
test(args, model, device, test_loader)


