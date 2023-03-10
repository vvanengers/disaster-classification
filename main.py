import argparse
import copy
import time

import numpy as np
import torch

from utils.checkpointer import Checkpointer
from utils.dataloader import load_folded_dataloaders, load_data_from_folder
from utils.models import initialize_model
from utils.other import print_and_log, setup_logger
from copy import deepcopy



def train_model(device, model, criterion, optimizer, scheduler, train_data_loader, val_data_loader, num_epochs,
                start_epoch):
    # start timer
    since = time.time()

    # best model baseline
    best_model_wts = copy.deepcopy(model.state_dict())
    # best accuracy baseline
    best_acc = 0.0
    best_preds = []
    best_labels = []
    best_paths = []
    hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in np.arange(start_epoch, num_epochs, 1):
        print_and_log(f'Epoch {epoch}/{num_epochs - 1}')
        print_and_log(f'Learning rate: {scheduler.get_last_lr()}')
        print_and_log('-' * 10)

        # Each epoch has a training and validation phase

        # training phase
        train_loss, train_acc, _, __, ___ = do_epoch('train', train_data_loader, model, criterion, optimizer, scheduler,
                                                      device)
        hist['train_loss'].append(train_loss)
        hist['train_acc'].append(train_acc)

        # validation phase
        val_loss, val_acc, val_preds, val_labels, val_paths = do_epoch('val', val_data_loader, model, criterion,
                                                                       optimizer, scheduler, device,)
        hist['val_loss'].append(val_loss)
        hist['val_acc'].append(val_acc)

        # save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_preds = val_preds
            best_labels = val_labels
            best_paths = val_paths

    # get time
    time_elapsed = time.time() - since
    print_and_log(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print_and_log(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_model_wts, best_acc, best_preds, best_labels, best_paths, hist


def do_epoch(phase, dataloader, model, criterion, optimizer, scheduler, device):
    print_and_log(f'Start {phase} phase')
    if phase == 'train':
        model.train()  # Set model to training mode
    else:
        model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    all_paths = []

    # Iterate over data.
    for i, (inputs, labels, paths) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            # layer freezing
            # for layer in [model.layer1, model.layer2]:
            #     for param in layer.parameters():
            #         param.requires_grad = False
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
        all_preds += preds
        all_labels += labels.data
        all_paths += paths
    if phase == 'train':
        scheduler.step()
    epoch_loss = running_loss / len(all_preds)
    epoch_acc = running_corrects.double() / len(all_preds)
    print_and_log(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return epoch_loss, epoch_acc, all_preds, all_labels, all_paths


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='resnet18', help='Model to use.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of training and testing data.')
    parser.add_argument('--feature_extract', action='store_true', default=True)
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--epochs', type=int, default=64, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--stepsize', type=int, default=10, help='Stepsize.')
    parser.add_argument('--k_folds', type=int, default=5, help='Kfolds')
    parser.add_argument('--gamma', type=float, default=0.5, help='Reduction of lr.')
    parser.add_argument('--model_save_path', type=str, default='models/', help='Where to save the models.')
    parser.add_argument('--model_load_path', type=str, default=None, help='Path to model to load. No model is loaded'
                                                                          'if value is None')
    parser.add_argument('--result_save_path', type=str, default='results/', help='Where to save the results.')
    parser.add_argument('--dataset_path', type=str, default='data/Incidents-subset',
                        help='Where to load the dataset from')
    args = parser.parse_args()

    setup_logger(args)
    print_and_log(args)

    # setup checkpointing
    # set save name to current time
    save_name = time.strftime("%Y%m%d%H%M%S") + args.model
    model_checkpointer = Checkpointer(args.model_save_path, save_name, args, autosave=False)
    result_checkpointer = Checkpointer(args.result_save_path, save_name, args, autosave=False)

    # get dataset
    dataset = load_data_from_folder(args.dataset_path)

    # set the device to cuda if gpu is available, otherwise use cpu
    d = "cuda:0" if torch.cuda.is_available() else "cpu"
    print_and_log(f'Training with {d}')
    device = torch.device(d)

    # setup model
    num_classes = len(dataset.classes)
    model, input_size = initialize_model(args.model, num_classes, args.feature_extract, args.pretrained)

    # Send the model to GPU
    model = model.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model.parameters()
    print("Params to learn:")
    if args.feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # setup criterion with class weighting
    criterion = torch.nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.Adam(params_to_update, lr=args.lr)

    # setup learning rate scheduler
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=args.stepsize, gamma=args.gamma)

    # load previous model from checkpoint if path is given
    start_epoch = 0
    if args.model_load_path:
        print_and_log(f'Loading model from {args.load_model_path}')
        checkpoint = torch.load(args.load_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss_hist = checkpoint['loss_hist']
        train_hist = checkpoint['train_hist']
        val_hist = checkpoint['val_hist']

    orig_model_state_dict = copy.deepcopy(model.state_dict())
    orig_optmizer_state_dict = copy.deepcopy(optimizer_ft.state_dict())
    if args.train:
        folded_best_acc = 0
        folded_best_model_wts = None
        folded_data_loaders = load_folded_dataloaders(dataset, k_folds=args.k_folds)
        for k, (train_loader, valid_loader) in enumerate(folded_data_loaders):
            print_and_log(f'Started training fold {k}')
            model.load_state_dict(copy.deepcopy(orig_model_state_dict))
            optimizer_ft.load_state_dict(copy.deepcopy(orig_optmizer_state_dict))
            _, best_model_wts, best_acc, best_preds, best_labels, best_paths, hist = train_model(device, model,
                                                                                                 criterion,
                                                                                                 optimizer_ft,
                                                                                                 exp_lr_scheduler,
                                                                                                 train_loader,
                                                                                                 valid_loader,
                                                                                                 args.epochs,
                                                                                                 start_epoch)
            if best_acc > folded_best_acc:
                folded_best_acc = best_acc
                folded_best_model_wts = best_model_wts
            result_checkpointer.add_in_list('folded_hist', hist)
            result_checkpointer.add_in_list('folded_best_acc', best_acc)
            result_checkpointer.add_in_list('folded_best_preds', best_preds)
            result_checkpointer.add_in_list('folded_best_labels', best_labels)
            result_checkpointer.add_in_list('folded_best_paths', best_paths)
        model_checkpointer.add_singular('folded_best_model_wts', folded_best_model_wts)
        result_checkpointer.save()
        model_checkpointer.save()


if __name__ == '__main__':
    main()
