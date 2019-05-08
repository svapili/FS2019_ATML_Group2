import csv
import glob
import os

import matplotlib.pyplot as plt

my_path = os.path.abspath(os.path.dirname(__file__))
dir = os.path.dirname(my_path)
results_dir = dir + '/results'

files = glob.glob(results_dir + "/*.csv", recursive=True)

for f in files:
    file_name = f.rsplit('/')[-1]
    parameters = file_name.split('_')
    netname = parameters[0]
    loss_fn = parameters[1]
    lr = parameters[2]
    optimizer = parameters[3]
    with open(f) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        list_training = []
        for row in csv_reader:
            list_training.append(row)
        train_losses = list_training[0]
        train_accuracies = list_training[1]
        val_losses = list_training[2]
        val_accuracies = list_training[3]
        epochs = list(range(1,len(val_accuracies)))

    plt.plot(epochs,train_accuracies[1::])
    plt.show()