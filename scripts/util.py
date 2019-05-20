import os
import csv
import glob
import matplotlib.pyplot as plt
import torch

'''
Show the given input image
inp:    the image to show
title:  the title to display
'''
def imshow(inp, title=None):
    """Imshow for Tensor."""
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


'''
A function for saving results
Args
epoch:  the number of epochs
config: the model used
loss_fn: the loss function used
...
'''
def save_results(epoch, config, loss_fn, learning_rate, optimizer, path_idx, results_dir, modelstate_dir, 
                 pretrained, model_idx, model, train_losses, train_accuracies, val_losses, val_accuracies, learn_rates,
                 time_epoch, TPs, TNs, FPs, FNs):
            
    print('...saving...')
    name = str(pretrained) + '_' + str(path_idx) + str(model_idx) + '_' + config + '_' + loss_fn.__str__() + '_lr=' + str(learning_rate) + '_' +(optimizer.__str__()).split(' ')[0]

    #remove old results
    for filename in glob.glob(results_dir + '/' + name + '*'):
        os.remove(filename)
    for filename in glob.glob(modelstate_dir + '/' + name + '*'):
        os.remove(filename)

    name = name + '_Epoch_' + str(epoch+1)

    # save model weights
    torch.save(model.state_dict(), modelstate_dir + '/' + name + '.pth')

    # save results per epoch
    path = results_dir + '/' + name + '.csv'
    with open(path, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(train_losses)
        writer.writerow(train_accuracies)
        writer.writerow(val_losses)
        writer.writerow(val_accuracies)
        writer.writerow(learn_rates)
        writer.writerow(time_epoch)
        writer.writerow(TPs)
        writer.writerow(TNs)
        writer.writerow(FPs)
        writer.writerow(FNs)
    csvFile.close()
            