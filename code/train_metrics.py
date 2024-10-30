from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import torch


def intersection(im1,im2):
    im1 = (im1 > 0)
    inter = im1*im2
    return inter

def union(im1,im2):
    im1 = (im1>0)
    im2 = (im2>0)
    uni = (im1+im2)*1.0 
    return uni

def image_sum(im1,im2):
    im1 = (im1>0)*1.0
    im_sum = im1+im2
    return im_sum

def dice_coeff(pred, target):
    return 2 * torch.sum(intersection(pred,target)) / torch.sum(image_sum(pred,target))

def iou_coeff(pred, target):
    return torch.sum(intersection(pred,target))/torch.sum(union(pred,target))

def plot_loss(train_loss, val_loss):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train loss', color='tab:green')
    ax1.plot(train_loss, color='tab:green')
    ax1.tick_params(axis='y', labelcolor='tab:green')
    plt.xticks(range(len(train_loss))) # integer x-axis
    ax2 = ax1.twinx()  

    ax2.set_ylabel('Val loss', color= 'tab:blue') 
    ax2.plot(val_loss, color= 'tab:blue')
    ax2.tick_params(axis='y', labelcolor= 'tab:blue')

    fig.suptitle('Training and Validation Loss')
    fig.tight_layout()  
    plt.show()

def plot_dice_coeff(train_dice_coeff, val_dice_coeff):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train dice', color='tab:green')
    ax1.plot(train_dice_coeff, color='tab:green')
    ax1.tick_params(axis='y', labelcolor='tab:green')
    plt.xticks(range(len(train_dice_coeff))) 
    ax2 = ax1.twinx()  

    ax2.set_ylabel('Val dice', color= 'tab:blue') 
    ax2.plot(val_dice_coeff, color= 'tab:blue')
    ax2.tick_params(axis='y', labelcolor= 'tab:blue')

    fig.suptitle('Training and Validation Dice')
    fig.tight_layout()  
    plt.show()

def plot_iou_coeff(train_iou_coeff, val_iou_coeff):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train IoU', color='tab:green')
    ax1.plot(train_iou_coeff, color='tab:green')
    ax1.tick_params(axis='y', labelcolor='tab:green')
    plt.xticks(range(len(train_iou_coeff))) 
    ax2 = ax1.twinx()  

    ax2.set_ylabel('Val IoU', color= 'tab:blue') 
    ax2.plot(val_iou_coeff, color= 'tab:blue')
    ax2.tick_params(axis='y', labelcolor= 'tab:blue')

    fig.suptitle('Training and Validation IoU')
    fig.tight_layout()  
    plt.show()

def plot_metrics(metrics):
    fig, axs = plt.subplots(1,3, figsize = (14,4))

    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].plot(metrics["train_loss"], color='tab:green', label = 'Train')
    axs[0].plot(metrics["val_loss"], color='tab:blue', label = 'Val')
    axs[0].legend(loc="upper right")
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True)) # integer only x-axis
    axs[0].set_title('Loss')

    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Dice')
    axs[1].plot(metrics["train_dice"], color='tab:green', label = 'Train')
    axs[1].plot(metrics["val_dice"], color='tab:blue', label = 'Val')
    axs[1].legend(loc="lower right")
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].set_title('Dice')

    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('IoU')
    axs[2].plot(metrics["train_iou"], color='tab:green', label = 'Train')
    axs[2].plot(metrics["val_iou"], color='tab:blue', label = 'Val')
    axs[2].legend(loc="lower right")
    axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].set_title('IoU')

    fig.suptitle('Training and Validation metrics')
    fig.tight_layout()  
    plt.show()

