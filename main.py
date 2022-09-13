from model import Net
import torch
import utils
import torch.nn as nn
import torch.optim as optim
from train import train
from test import test






if __name__ == '__main__':
    classes = ('clear','people')
    train_set,test_set=utils.create_dataset()
    training_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #Папка для сохранения весов
    PATH = './my_net.pth'

    print("Enter Mode: (train or test)")
    mode=input()

    if mode=='train':
        train(net,training_loader,classes,optimizer,criterion,PATH)
    elif mode=='test':
        test(net,validation_loader,PATH,classes)
    else:
        print("Wrong mode")