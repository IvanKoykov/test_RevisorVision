import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np




# функция для показа изображения
def imshow(img):
    img = img / 2 + 0.5     # денормализуем
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def data_processing():
    img_size = 64
    transform = transforms.Compose(
        [transforms.Resize((img_size, img_size)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform

def create_dataset():
    transform=data_processing()
    dataset = torchvision.datasets.ImageFolder(root='images', transform=transform)
    print(dataset.class_to_idx)
    print('Training set has {} instances'.format(len(dataset)))
    train_set, test_set = torch.utils.data.random_split(dataset, [27683, 10000],
                                                        generator=torch.Generator().manual_seed(42))
    return train_set,test_set
