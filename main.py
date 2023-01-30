import argparse
import copy
import time
import numpy as np
import torch
from torch import device
from torchvision import models

from dataloader import load_data
from utils import save


def checkpoint(path, name, epoch, model_state_dict, optimizer_state_dict, loss_hist, accuracy_hist):
    obj = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'loss_hist': loss_hist,
        'accuracy_hist': accuracy_hist
    }
    save(obj, path, name)


def train_model(device, model, criterion, optimizer, scheduler, dataloaders, save_path, save_name, num_epochs, start_epoch,
                accuracy_hist, loss_hist):
    loss_list = []
    acc_list = []
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        # for phase in ['train', 'val']:
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                # print(f'{i}/{len(dataloaders[phase])}')

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / len(dataloaders[phase])
            loss_list.append(epoch_loss)
            acc_list.append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        checkpoint(save_path, save_name, epoch, model.state_dict(), optimizer.state_dict(), loss_list, acc_list)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    checkpoint(save_path, save_name, num_epochs, model.state_dict(), optimizer.state_dict(), loss_list, acc_list)
    return model, loss_list, acc_list


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size in train-test split.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of training and testing data.')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=64, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--stepsize', type=int, default=7, help='Stepsize.')
    parser.add_argument('--model_save_path', type=str, default='models/', help='Where to save the models.')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to model to load. No model is loaded'
                                                                          'if value is None')
    args = parser.parse_args()

    train_batches, test_batches, sample_dist, names = load_data('data/Incidents-subset', test_size=args.test_size,
                                                                batch_size=args.batch_size)

    dataloaders = {'train': train_batches}

    # set the device to cuda if gpu is available, otherwise use cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet18(pretrained=args.pretrained)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = torch.nn.Linear(num_ftrs, len(names))

    model_ft = model_ft.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=args.lr, momentum=args.momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=args.stepsize, gamma=0.1)


    # load previous model from checkpoint if path is given
    accuracy_hist, loss_hist = [], []
    start_epoch = 0
    if args.load_model_path:
        checkpoint = torch.load(args.load_model_path)
        model_ft.load_state_dict(checkpoint['model_state_dict'])
        optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss_hist = checkpoint['loss_hist']
        accuracy_hist = checkpoint['accuracy_hist']


    # set model save name to current time
    model_save_name = time.strftime("%Y%m%d%H%M%S")
    best_model, loss_list, acc_list = train_model(device, model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders,
                           args.model_save_path, model_save_name, args.epochs, start_epoch, accuracy_hist,
                                                  loss_hist)


if __name__ == '__main__':
    main()
