import numpy as np
import matplotlib.pyplot as plt


def showSamples(dataset):

    plt.figure()

    for i in range(0, 5):
        index = np.random.choice(dataset.__len__())
        plt.subplot(1, 5, i+1)
        try:
            plt.title(dataset[index]['classification'])
            plt.imshow(dataset[index]['image'])
        except Exception as e:
            print("Error with image " + np.str(index) + ": " + e.args[0])
        
