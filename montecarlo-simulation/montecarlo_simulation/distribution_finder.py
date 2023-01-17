import argparse
from pprint import pprint

from fitter import Fitter

import numpy as np
np.set_printoptions(precision=3)

from montecarlo_simulation.data_loader import DataLoader

def fit(dataframe, column, print_summary=True):
    data = dataframe[column].to_numpy()
    fitter = Fitter(data)

    fitter.fit()

    if print_summary:
        fitter.summary()

    return fitter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("column")

    args = parser.parse_args()

    dataloader = DataLoader(args.filename, columns=[args.column])

    result = fit(dataloader.data, args.column)

    pprint(result)