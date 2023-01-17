import numpy as np
import matplotlib.pyplot as plt

def calculate_inundation(flood_depth,first_floor_height):
    inundation=flood_depth - first_floor_height

    return inundation

def CDF_emprical_distribution(data,color="gray",number_bins=100,label="Nested Monte Carlo Simulation ",show=True):
    count, bins_count = np.histogram(data, bins=number_bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], cdf, label=label,color=color, linestyle='solid',linewidth=0.4)

    if label:
        plt.legend()

    if show:
        plt.show()











