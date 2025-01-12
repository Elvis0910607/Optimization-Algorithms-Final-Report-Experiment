import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torchvision import transforms, datasets 
from torch.utils.data.sampler import Sampler

def MNIST_dataset():
    if not os.path.isdir("data"):
        os.mkdir("data")
    # Download MNIST dataset and set the valset as the test test
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    test_set = datasets.MNIST('data/MNIST', download=True, train=False, transform=transform)
    train_set = datasets.MNIST("data/MNIST", download=True, train=True, transform=transform)
    return train_set, test_set

def MNIST_nn_one_layer():
    model = nn.Sequential(
        nn.Linear(784, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.LogSoftmax(dim = 1)
    )
    return model

def MNIST_logistic():
    model = nn.Sequential(
        nn.Linear(784, 10),
        nn.LogSoftmax(dim = 1)
    )
    return model

def accuracy(yhat, labels):
    _, indices = yhat.max(1)
    return (indices == labels).sum().data.item() / float(len(labels))

    
class AverageCalculator():
    def __init__(self):
        self.reset() 
    
    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
    
    def update(self, val, n=1):
        assert(n > 0)
        self.sum += val * n 
        self.count += n
        self.avg = self.sum / float(self.count)

def plot_train_stats(train_loss, val_loss, train_acc, val_acc, directory, acc_low=0):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6), sharey='row')
    axes[0][0].plot(np.array(train_loss))
    axes[0][0].set_title("Training Loss")
    axes[0][1].plot(np.array(val_loss))
    axes[0][1].set_title("Validation Loss")
    axes[1][0].plot(np.array(train_acc))
    axes[1][0].set_title("Training Accuracy")
    axes[1][0].set_ylim(acc_low, 1)
    axes[1][1].plot(np.array(val_acc))
    axes[1][1].set_title("Validation Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'train_stats.png'))
    plt.show() 
    
def plot_comparison(train_stats_sgd, train_stats_svrg, train_stats_storc, directory, acc_low=0):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), sharey='row')
    
    # Extract stats
    train_loss_sgd, val_loss_sgd, train_acc_sgd, val_acc_sgd = train_stats_sgd
    train_loss_svrg, val_loss_svrg, train_acc_svrg, val_acc_svrg = train_stats_svrg
    train_loss_storc, val_loss_storc, train_acc_storc, val_acc_storc = train_stats_storc

    # Loss Curves
    axes[0][0].plot(train_loss_sgd, label="SGD", color='blue')
    axes[0][0].plot(train_loss_svrg, label="SVRG", color='red')
    axes[0][0].plot(train_loss_storc, label="STORC", color='green')
    axes[0][0].set_title("Training Loss")
    axes[0][0].legend()

    axes[0][1].plot(val_loss_sgd, label="SGD", color='blue')
    axes[0][1].plot(val_loss_svrg, label="SVRG", color='red')
    axes[0][1].plot(val_loss_storc, label="STORC", color='green')
    axes[0][1].set_title("Validation Loss")
    axes[0][1].legend()

    # Accuracy Curves
    axes[1][0].plot(train_acc_sgd, label="SGD", color='blue')
    axes[1][0].plot(train_acc_svrg, label="SVRG", color='red')
    axes[1][0].plot(train_acc_storc, label="STORC", color='green')
    axes[1][0].set_title("Training Accuracy")
    axes[1][0].set_ylim(acc_low, 1)
    axes[1][0].legend()

    axes[1][1].plot(val_acc_sgd, label="SGD", color='blue')
    axes[1][1].plot(val_acc_svrg, label="SVRG", color='red')
    axes[1][1].plot(val_acc_storc, label="STORC", color='green')
    axes[1][1].set_title("Validation Accuracy")
    axes[1][1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'optimizer_comparison.png'))
    plt.show()
