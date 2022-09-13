import torchvision
import utils
import torch



def train(net,training_loader,classes,optimizer,criterion,PATH):
    # получаем несколько случайных тренировочных изображений
    dataiter = iter(training_loader)
    images, labels = dataiter.next()

    # показываем изображения
    utils.imshow(torchvision.utils.make_grid(images))
    # печатаем метки
    print(' '.join(classes[labels[j]] for j in range(4)))
    print('Start Train')



    for epoch in range(20):

        running_loss = 0.0
        for i, data in enumerate(training_loader, 0):

            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            # print(labels," ", outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), PATH)
    print("Weights Saved")
    return PATH
