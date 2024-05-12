import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import torch
from albumentations import Compose, PadIfNeeded, RandomCrop, Normalize, HorizontalFlip, VerticalFlip, ShiftScaleRotate, Cutout, CoarseDropout
from albumentations.pytorch.transforms import ToTensorV2

from pytorch_grad_cam.base_cam import BaseCAM

import torchvision
from torchvision import datasets

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pl_bolts.datamodules import CIFAR10DataModule

import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix


import os
from tqdm import tqdm

# Image Transformation
# Data
print('==> Preparing data..')
class album_Compose_train():
    def __init__(self):
        self.albumentations_transform = Compose([
                            RandomCrop (height = 32, width = 32, always_apply=False, p=4.0),
                            HorizontalFlip(),
                            #Cutout(num_holes=1, max_h_size=8, max_w_size=8, always_apply=True),
                            CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=[0.4914*255, 0.4822*255, 0.4471*255], mask_fill_value=None),
                            Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
                            ToTensorV2()])
    def __call__(self,img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img

class album_Compose_test():
    def __init__(self):
        self.albumentations_transform = Compose([#  transforms.Resize((28, 28)),
                           #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                           Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
                           ToTensorV2()
                           ])

    def __call__(self,img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img
    

class dataset_cifar10:
    def __init__(self, batch_size):
        
        SEED = 1

        # CUDA?
        cuda = torch.cuda.is_available()
        print("CUDA Available?", cuda)

        # For reproducibility
        torch.manual_seed(SEED)
        
        if cuda:
            torch.cuda.manual_seed(SEED)

        # dataloader arguments - something you'll fetch these from cmdprmt
        self.dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=128)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
        
    def train_test_datasets(self):
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=album_Compose_train())
        trainloader = torch.utils.data.DataLoader(trainset, **self.dataloader_args)

        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=album_Compose_test())
        testloader = torch.utils.data.DataLoader(testset, **self.dataloader_args)
        
        return(trainloader, testloader)

    #Display Images   
    def sample_pictures(self, data_loader, train_flag = True, return_flag = False):
        dataiter = iter(data_loader)
        images, labels = next(dataiter)
        images = images.numpy() # convert images to numpy for display

        def imshow(img):
            channel_means = [0.4914, 0.4822, 0.4471]
            channel_std = [0.2469, 0.2433, 0.2615]

            for i in range(img.shape[0]):
                img[i]=(img[i]*channel_std[i])+channel_means[i]
                plt.imshow(np.transpose(img, (1,2,0)))

        fig = plt.figure(figsize=(25, 25))
        fig.tight_layout()
        num_of_images = 25 if train_flag else 5
        for index in range(1, num_of_images + 1):
            ax = fig.add_subplot(5, 5, index, xticks=[], yticks=[])
            imshow(images[index])
            ax.set_title(self.classes[labels[index]])


def unnormalize(img):
    channel_means = (0.4914, 0.4822, 0.4471)
    channel_stdevs = (0.2469, 0.2433, 0.2615)
    img = img.numpy().astype(dtype=np.float32)
    
    for i in range(img.shape[0]):
        img[i] = (img[i]*channel_stdevs[i])+channel_means[i]
    return np.transpose(img, (1,2,0))

def plot_loss_accuracy_graph(trainObj, testObj, EPOCHS):

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    train_losses = [temp.cpu().detach() for temp in trainObj.train_losses]

    train_epoch_linspace = np.linspace(1, EPOCHS, len(train_losses))
    test_epoch_linspace = np.linspace(1, EPOCHS, len(testObj.test_losses))

    # Loss Plot
    ax[0].plot(train_epoch_linspace, train_losses, label='Training Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss vs. Epochs')
    ax[0].legend()

    ax2 = ax[0].twinx()
    ax2.plot(test_epoch_linspace, testObj.test_losses, label='Test Loss', color='red')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='center right')

    # Accuracy Plot
    #train_acc = [temp.cpu().detach() for temp in trainObj.train_acc]
    ax[1].plot(train_epoch_linspace, trainObj.train_acc, label='Training Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy vs. Epochs')
    ax[1].legend()

    ax2 = ax[1].twinx()
    ax2.plot(test_epoch_linspace, testObj.test_acc, label='Test Accuracy', color='red')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='center right')

    plt.tight_layout()
    plt.show()

def plot_loss_accuracy_graph_OneCLR(trainAcc, trainLoss, testAcc, testLoss):

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Loss Plot
    ax[0].plot(trainLoss, label='Training Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss vs. Epochs')
    ax[0].legend()

    ax2 = ax[0].twinx()
    ax2.plot(testLoss, label='Test Loss', color='red')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='center right')

    # Accuracy Plot
    ax[1].plot(trainAcc, label='Training Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy vs. Epochs')
    ax[1].legend()

    ax2 = ax[1].twinx()
    ax2.plot(testAcc, label='Test Accuracy', color='red')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='center right')

    plt.tight_layout()
    plt.show()  

# Evaluating Train and Test Accuracy
import torch

def calAccuracy(net, dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images,labels = images.to(device),labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the  train images: {(100 * correct / total)} %%')

def calClassAccuracy(net, dataloader, classes, device):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images,labels = images.to(device),labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    
# Define a function to plot misclassified images
def plot_misclassified_images(model, test_loader, classes, device):
    # set model to evaluation mode
    model.eval()

    misclassified_images = []
    actual_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    misclassified_images.append(data[i])
                    actual_labels.append(classes[target[i]])
                    predicted_labels.append(classes[pred[i]])

    # Plot the misclassified images
    fig = plt.figure(figsize=(12, 5))
    for i in range(10):
        sub = fig.add_subplot(2, 5, i+1)
        npimg = unnormalize(misclassified_images[i].cpu())
        plt.imshow(npimg, cmap='gray', interpolation='none')
        sub.set_title("Actual: {}, Pred: {}".format(actual_labels[i], predicted_labels[i]),color='red')
    plt.tight_layout()
    plt.show()

def plot_grad_cam_images_custom_resnet(model, test_loader, classes, device):
    # set model to evaluation mode
    model.eval()
    target_layers = [model.R3]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)

    misclassified_images = []
    actual_labels = []
    actual_targets = []
    predicted_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    actual_targets.append(target[i])
                    misclassified_images.append(data[i])
                    actual_labels.append(classes[target[i]])
                    predicted_labels.append(classes[pred[i]])

    # Plot the misclassified images
    fig = plt.figure(figsize=(12, 5))
    for i in range(10):
        sub = fig.add_subplot(2, 5, i+1)
        input_tensor = misclassified_images[i].unsqueeze(dim=0)
        targets = [ClassifierOutputTarget(actual_targets[i])]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(unnormalize(misclassified_images[i].cpu()), grayscale_cam, use_rgb=True, image_weight=0.7)

        # npimg = unnormalize(misclassified_images[i].cpu())
        # plt.imshow(npimg, cmap='gray', interpolation='none')

        # npimg = unnormalize(misclassified_images[i].cpu())
        plt.imshow(visualization)
        sub.set_title("Actual: {}, Pred: {}".format(actual_labels[i], predicted_labels[i]),color='red')
    plt.tight_layout()
    plt.show()

def plot_grad_cam_images(model, test_loader, classes, device):
    # set model to evaluation mode
    model.eval()
    target_layers = [model.layer4[-1]]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)

    misclassified_images = []
    actual_labels = []
    actual_targets = []
    predicted_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    actual_targets.append(target[i])
                    misclassified_images.append(data[i])
                    actual_labels.append(classes[target[i]])
                    predicted_labels.append(classes[pred[i]])

    # Plot the misclassified images
    fig = plt.figure(figsize=(12, 5))
    for i in range(10):
        sub = fig.add_subplot(2, 5, i+1)
        input_tensor = misclassified_images[i].unsqueeze(dim=0)
        targets = [ClassifierOutputTarget(actual_targets[i])]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(unnormalize(misclassified_images[i].cpu()), grayscale_cam, use_rgb=True, image_weight=0.7)

        # npimg = unnormalize(misclassified_images[i].cpu())
        # plt.imshow(npimg, cmap='gray', interpolation='none')

        # npimg = unnormalize(misclassified_images[i].cpu())
        plt.imshow(visualization)
        sub.set_title("Actual: {}, Pred: {}".format(actual_labels[i], predicted_labels[i]),color='red')
    plt.tight_layout()
    plt.show()

    # ---------------------------- Confusion Matrix ----------------------------
def visualize_confusion_matrix(classes: list[str], device: str, model: 'DL Model',
                               test_loader: torch.utils.data.DataLoader):
    """
    Function to generate and visualize confusion matrix
    :param classes: List of class names
    :param device: cuda/cpu
    :param model: Model Architecture
    :param test_loader: DataLoader for test set
    """
    nb_classes = len(classes)
    device = 'cuda'
    cm = torch.zeros(nb_classes, nb_classes)

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            model = model.to(device)

            preds = model(inputs)
            preds = preds.argmax(dim=1)

        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[t, p] = cm[t, p] + 1

    # Build confusion matrix
    labels = labels.to('cpu')
    preds = preds.to('cpu')
    cf_matrix = confusion_matrix(labels, preds)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
                         index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)



