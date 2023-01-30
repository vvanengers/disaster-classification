import argparse
import copy
import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import models as tmodels

from dataloader import load_data
from utils import save

logger = None

models = {
    'resnet34': tmodels.resnet34,
    'resnet18': tmodels.resnet18
}


def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    Path('logs/').mkdir(parents=True, exist_ok=True)
    log_path = f'logs/{time.strftime("%Y%m%d%H%M%S")}'

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)


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
        print_and_log(f'Epoch {epoch}/{num_epochs - 1}')
        print_and_log('-' * 10)

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
                    print_and_log(f'model: {str(next(model.parameters()).device)}')
                    print_and_log(f'inputs: {str(inputs.device)}')
                    outputs = model(inputs)
                    print_and_log(f'outputs: {str(outputs.device)}. labels: {str(labels.device)}')
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

            print_and_log(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        checkpoint(save_path, save_name, epoch, model.state_dict(), optimizer.state_dict(), loss_list, acc_list)

    time_elapsed = time.time() - since
    print_and_log(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print_and_log(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    checkpoint(save_path, save_name, num_epochs, model.state_dict(), optimizer.state_dict(), loss_list, acc_list)
    return model, loss_list, acc_list


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='resnet18', help='Model to use.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size in train-test split.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of training and testing data.')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--epochs', type=int, default=64, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--stepsize', type=int, default=7, help='Stepsize.')
    parser.add_argument('--model_save_path', type=str, default='models/', help='Where to save the models.')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to model to load. No model is loaded'
                                                                          'if value is None')
    args = parser.parse_args()

    setup_logger(args)
    print_and_log(args)

    train_batches, test_batches, sample_dist, names = load_data('data/Incidents-subset', test_size=args.test_size,
                                                                batch_size=args.batch_size)

    dataloaders = {'train': train_batches}

    # set the device to cuda if gpu is available, otherwise use cpu
    d = "cuda:0" if torch.cuda.is_available() else "cpu"
    print_and_log(f'Training with {d}')
    device = torch.device(d)

    if args.model not in models:
        raise AttributeError(f'Model {args.model} unknown. ')

    model_ft = models[args.model](pretrained=args.pretrained).to(device)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = torch.nn.Linear(num_ftrs, len(names))

    model_ft = model_ft.to(device)

    weight = torch.tensor(sample_dist) / np.sum(sample_dist)
    criterion = torch.nn.CrossEntropyLoss(weight=weight).to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=args.lr, momentum=args.momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=args.stepsize, gamma=0.1)


    # load previous model from checkpoint if path is given
    accuracy_hist, loss_hist = [], []
    start_epoch = 0
    if args.load_model_path:
        print_and_log(f'Loading model from {args.load_model_path}')
        checkpoint = torch.load(args.load_model_path)
        model_ft.load_state_dict(checkpoint['model_state_dict'])
        optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss_hist = checkpoint['loss_hist']
        accuracy_hist = checkpoint['accuracy_hist']

    # set model save name to current time
    model_save_name = time.strftime("%Y%m%d%H%M%S") + args.model
    best_model, loss_list, acc_list = train_model(device, model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders,
                           args.model_save_path, model_save_name, args.epochs, start_epoch, accuracy_hist,
                                                  loss_hist)


if __name__ == '__main__':
    main()
