import numpy as np
import matplotlib.pyplot as plt

        
def imshow(inp, title=None):
    """Imshow for Tensor."""
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    