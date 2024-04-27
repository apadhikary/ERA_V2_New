'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms

from torch.optim.lr_scheduler import StepLR

import os
from tqdm import tqdm


# Training
class train:
    def __init__(self):
        self.train_losses = []
        self.train_acc = []

    def train(self,model, device, train_loader, optimizer, epoch):
        model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(device), target.to(device)

            # Init
            optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred = model(data)

            # Calculate loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(y_pred, target)
            self.train_losses.append(loss)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update pbar-tqdm

            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)
        
        return (loss, correct, processed)

#Testing
class test:
    def __init__(self):
        self.test_acc = []
        self.test_losses = []

    def test(self,model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                criterion = nn.CrossEntropyLoss()
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        self.test_acc.append(100. * correct / len(test_loader.dataset))

        return (test_loss,correct)

class get_lr:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        """
        Function to track learning rate while model training
        :param optimizer: Optimizer used for training
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
