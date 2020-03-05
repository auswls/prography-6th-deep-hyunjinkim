import argparse
import os
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable

from model import VGG16

def train(args, model, device, train_loader, optimizer, epoch):
   """Training"""
   print("training - epoch : ", epoch)
   model.train()
   for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      output = model(data)
      criterion = nn.CrossEntropyLoss()
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      if batch_idx % args.log_interval == 0:
         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
         #print('{{"metric": "Train - NLL Loss", "value": {}}}'.format(loss.item()))


def test(args, model, device, test_loader, epoch):
   """Testing"""
   print("testing - epoch : ", epoch)
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


def main():
   # Training settings
   parser = argparse.ArgumentParser(description='Prography-6th-assignment-HyunjinKim')
   parser.add_argument('--dataroot', default="/input/" ,help='path to dataset')
   parser.add_argument('--evalf', default="/eval/" ,help='path to evaluate sample')
   parser.add_argument('--outf', default='models',
           help='folder to output images and model checkpoints')
   parser.add_argument('--ckpf', default='',
           help="path to model checkpoint file (to continue training)")
           
   #### Batch size ####
   parser.add_argument('--batch-size', type=int, default=4, metavar='N',
           help='input batch size for training (default: 4)')
   parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
           help='input batch size for testing (default: 4)')
   #### Epochs ####
   parser.add_argument('--epochs', type=int, default=10, metavar='N',
           help='number of epochs to train (default: 10)')

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

   # use CUDA?
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


   # MNIST Dataset(for training)
   train_dataset = datasets.MNIST(root='./data/',
                            train=True,
                            transform=rgb_tranform,
                            download=True)
   train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


   # MNIST Dataset(for test)
   test_dataset = datasets.MNIST(root='./data/',
                           train=False,
                           transform=rgb_tranform)

   test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

   model = VGG16().to(device)
   print("model : ", model)

   optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


   for epoch in range(1, args.epochs + 1) :
      train(args, model, device, train_loader, optimizer, epoch)
      test(args, model, device, test_loader, epoch)
      torch.save(model.state_dict(), '/content/drive/My Drive/prography/model/mnist_convnet_model_epoch_%d.pth' % (epoch))



if __name__ == '__main__':
   main()