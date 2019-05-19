# Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import csv
import glob
from copy import deepcopy

# Custom functions
import dataSplitter
import loader
import dataAugmenter
import SimpleNet
import train
import test_
import optimize
import util

import platform


if __name__ == '__main__':

#######################
# Config
#######################

    # Paths definitions
    cluster = False
    if cluster:
        Path = '/var/tmp/'
    else:
        Path = '../data/'
    dataDir = Path + 'ISIC-images'
    #trainDir = Path + 'ISIC-images/train/'
    #testDir = Path + 'ISIC-images/test/'
    #valDir = Path + 'ISIC-images/val/'
    
    deformedPath = Path+'ISIC-images_deformed'
    paddedPath = Path+'ISIC-images_padded'
    
    for path_idx, Path in enumerate([deformedPath, paddedPath]):
        dataDir= Path
        trainDir = Path + '/train/'
        testDir = Path + '/test/'
        valDir = Path + '/val/'
        print(trainDir)
    
        # Paths definitions for saving results and model state
        my_path = os.getcwd()
        dir = os.path.dirname(my_path)
        results_dir = dir + '/results'
        modelstate_dir = '/var/tmp/modelstate'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if not os.path.exists(modelstate_dir):
            os.makedirs(modelstate_dir)

        # Data pre-processing
        # Data have first to be downloaded with data_downloader.py and extracted with data_extractor.py
        newDataSplit = False # Set to true to split the data randomly again
        dataPreprocessing = False # Set to true to resize and augment the data

        # Check if we can use CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #######################
    # Pre-processing
    #######################

        # Performs a new random split of the data
        if (newDataSplit):
            testRatio = .1
            valRatio = .1
            dataSplitter.split(trainDir, testDir, valDir, testRatio, valRatio)

        # Preprocessing the data (resizing and augmenting)
        if (dataPreprocessing):
            dataAugmenter.preprocessData([trainDir, testDir, valDir], outSize=(300,300), keepAspectRatio=False)

    #######################
    # Data loading
    #######################

        batch_size = 8

        # Create dataset and dataloaders objects
        image_datasets, dataloaders = loader.melanomaDataLoader(dataDir, batch_size=batch_size)

        # Get dataset objects sizes
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test', 'val']}
        print("Size of the dataset objects: ", dataset_sizes)

        # Get the class names
        class_names = image_datasets['train'].classes
        print("Images class names: ", class_names)

        # Visualize sample images
        #print("Sample images:")
        #loader.showSample(dataloaders, dataset_sizes, class_names)



    #######################
    # Network configuration
    #######################

        # Model setup
        pretrained = False

        #model = models.alexnet(pretrained=True)
        #model = models.AlexNet(num_classes=2)
        model = models.resnet18(pretrained=pretrained) 
        #model = models.resnet50(pretrained=pretrained)

        n_epochs = 30

        for pretrained in [True, False]:
            for model_idx, model in enumerate([models.resnet18(pretrained=pretrained), models.resnet50(pretrained=pretrained)]):
                for optimizer_name in ['Adam', 'SGD']:
                    # Optimizer setup
                    finetune = True # Only has an effect if pretrained==True. If finetune==False, then fixed-feature
                    learning_rate = 0.001
                    #optimizer_name="Adam" # "Adam" or "SGD"

                    optimizer = optimize.setup_optimizer(model, learning_rate, optimizer_name, pretrained, finetune)

                    # Scheduler setup
                    schedule = True
                    step_size = 10
                    gamma = 0.1

                    if schedule:
                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
                    else:
                        scheduler = None

                    # Early stopping setup
                    earlyStop = True

                    if earlyStop:
                        best_val_loss = np.inf
                        best_model = None
                        max_epochs = 5 # if no improvement after 5 epochs, stop training
                        counter = 0

                    # Loss function setup
                    loss_fn = nn.CrossEntropyLoss()

                    # Other setup
                    debug_training_status = False
                    saving = True
                    model = model.to(device)


                ###############################
                # Training and saving results
                ###############################    

                    train_losses, train_accuracies = ['train_losses'], ['train_accuracies']
                    val_losses, val_accuracies = ['val_losses'], ['val_accuracies']
                    learn_rates = ['learning_rate']
                    time_epoch = ['execution time']

                    TPs = ['True Positives']
                    TNs = ['True Negatives']
                    FPs = ['False Positives']
                    FNs = ['False Negatives']

                    config  = model._get_name() + " " + "_bs=" + str(batch_size)

                    ##############################
                    # Training Epochs            #
                    ##############################

                    for epoch in range(n_epochs):

                        start_time_epoch = time.time()

                        train_loss, train_accuracy = train.train(model, dataloaders['train'], optimizer, loss_fn, device)
                        val_loss, val_accuracy, TP, TN, FP, FN  = test_.test(model, dataloaders['val'], loss_fn, device) 

                        train_losses.append(train_loss)
                        train_accuracies.append(train_accuracy)
                        val_losses.append(val_loss)
                        val_accuracies.append(val_accuracy)
                        TPs.append(TP)
                        TNs.append(TN)
                        FPs.append(FP)
                        FNs.append(FN)

                        # SCHEDULER
                        learn_rates.append(optimizer.param_groups[0]['lr'])
                        if scheduler:
                            scheduler.step()

                        # TIME CALCULATION
                        time_last_epoch = time.time() - start_time_epoch
                        time_epoch.append(time_last_epoch)

                        # OTHER METRICS


                        print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}, learn_rates: {:}, epoch execution time: {:.4f}'.format(
                            epoch + 1, n_epochs,
                            train_losses[-1],
                            train_accuracies[-1],
                            val_losses[-1],
                            val_accuracies[-1],
                            learn_rates[-1],
                            time_epoch[-1]))

                        print('True Positive: {}, True Negative: {}, False Positives: {}, False Negative: {}'.format(
                            TPs[-1],
                            TNs[-1],
                            FPs[-1],
                            FNs[-1]))

                    ##############################
                    # Saving results             #
                    ##############################

                        util.save_results(epoch, config, loss_fn, learning_rate, optimizer, path_idx, results_dir, modelstate_dir, 
                                 pretrained, model_idx, model, train_losses, train_accuracies, val_losses, val_accuracies, learn_rates,
                                 time_epoch, TPs, TNs, FPs, FNs)

                    ##############################
                    # Early stopping             #
                    ##############################
                        if earlyStop:
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_model = deepcopy(model)
                                counter = 0
                            else:
                                counter += 1
                            if counter == max_epochs:
                                print('No improvement for {} epochs; training stopped.'.format(max_epochs))

                                util.save_results(epoch, config, loss_fn, learning_rate, optimizer, path_idx, results_dir, modelstate_dir, 
                                 pretrained, model_idx, best_model, train_losses, train_accuracies, val_losses, val_accuracies, learn_rates,
                                 time_epoch, TPs, TNs, FPs, FNs)

                                break