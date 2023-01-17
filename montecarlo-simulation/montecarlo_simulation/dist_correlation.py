from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import argparse
import logging
import random
from functools import partial

from tqdm import tqdm

import numpy as np
import scipy.stats as sst
import matplotlib.pyplot as plt

from montecarlo_simulation.data_loader import DataLoader

logger = logging.getLogger("Dist Correlation")
NUM_SAMPLE = 5000
NUM_SIMULATION = 200

NUM_PROCESS = 1
NUM_THREAD = 1

def plot(x, list_of_y,x_raw=None,legend=None,line_style="-",show=True):
    if x_raw is not None:
        fig, ax = plt.subplots(1, 1)
        ax.hist(x_raw,density=True,bins=100)

    for y in list_of_y:
        plt.plot(x, y,line_style)

    if legend is not None:
        plt.xlabel(legend)
    if show:
        plt.show()

def get_distribution_model(data, model_class):
    args = model_class.fit(data)
    logger.debug("{} params".format(model_class.name), args)

    return model_class(*args)

def generate_data(data, model_class, num_sample):
    distribution_model = get_distribution_model(data, model_class)

    with ThreadPool(NUM_THREAD) as pool:
        result = pool.map(distribution_model.ppf, [random.random() for propapility_index in range(num_sample)])

    return np.array(result)

def generate_data_random(data, num_sample):
    return np.random.choice(data, size=num_sample)


def inundation_depth_simulation(flood_depth, first_floor_height, num_sample, calc_difference=True):
    dommy_flood_depth = generate_data_random(flood_depth, num_sample)
    dommy_first_floor_height = generate_data_random(first_floor_height, num_sample)


    result = {"flood_depth":dommy_flood_depth, "first_floor_height":dommy_first_floor_height}

    if calc_difference:
        difference = dommy_flood_depth - dommy_first_floor_height
        result['difference'] = difference

    return result

def inundation_depth_simulation_helper(kwargs, calc_difference):
    return inundation_depth_simulation(num_sample=NUM_SAMPLE, calc_difference=calc_difference, **kwargs)

def montecarlo_inundation_simulation(flood_depth, first_floor_height, num_simulations, num_levels=2):
    kwargs = {"flood_depth":flood_depth, "first_floor_height":first_floor_height}

    level = [kwargs for simulation_index in range(num_simulations)]
#    level=np.repeat(kwargs,num_simulations)
    with Pool(NUM_PROCESS) as pool:
        for level_index in range(1, num_levels+1):
            level = list(tqdm(pool.imap(partial(inundation_depth_simulation_helper,
                                                calc_difference=level_index==num_levels),
                                        level),
                              total=num_simulations**level_index,
                              desc="Calculating level {}".format(level_index)))

            if level_index != num_levels:
                level = np.repeat(level, num_simulations)

    return level

def calculate_correlation(data, model_class, degree):
    model = get_distribution_model(data, model_class)
    x = np.linspace(model.ppf(0.01), model.ppf(0.99), 100)
    y = model.pdf(x)

    fit_result = np.polyfit(x, y, deg=degree)

    ys = None
    for pow in range(0, degree+1):
        if pow == 0:
            ys = fit_result[degree-pow]
        else:
            ys += fit_result[degree-pow] * (x ** pow)

    plot(x, [y,ys],data,legend="Correlation is {} for degree {} for {} distribution ".format(np.corrcoef(y,ys)[0][1],degree, model_class.name))

    return np.corrcoef(y,ys)[0][1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument('-c', '--columns', help='Comma separated name of columns',
                        type=lambda arg: [int(column) for column in arg.split(',')], default=None)

    args = parser.parse_args()

    dataloader = DataLoader(args.filename, columns=args.columns)

    calculate_correlation(dataloader.data['Base_Fresh'], sst.bradford,degree=1)



