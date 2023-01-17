
import pandas as pd
import scipy as sp

class DataLoader:
    def __init__(self, file_path, columns=None):
        self.file_path = file_path
        self.columns = columns

        self.data = None

    def load(self):
        self.data = pd.read_csv(self.file_path, usecols=self.columns)


    def apply_rergression(self, column_x_idx, column_y_idx, offset=1):
        slope = []
        intercept = []

        for i in range(offset):
            slope.append(None)
            intercept.append(None)

        for i in range(offset, len(self.data)):
            x = self.data.iloc[i - offset:i + offset, column_x_idx]
            y = self.data.iloc[i - offset:i + offset, column_y_idx]
            s, i, r_value, p_value, std_err = sp.stats.linregress(x, y)
            slope.append(s)
            intercept.append(i)

        self.data["slope"] = slope
        self.data["intercept"] = intercept

