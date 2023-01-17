import pandas as pd
import matplotlib.pyplot as plt

def draw_pbox(upper_bound, lower_bound, xlim=None, xlabel=None, ylabel=None, size=18, fontsize=14, show_legend=True, show=True):

    upper_bound.loc[upper_bound.shape[0]] = [None]
    upper_bound["x"].iloc[200] = lower_bound["x"].iloc[199]

    lower_bound.loc[-1] = upper_bound.loc[0]

    lower_bound.index = lower_bound.index + 1

    lower_bound.sort_index(inplace=True)

    lower_bound = lower_bound.reset_index()

    lower_bound = lower_bound.rename(columns={'x': 'Upper Bound'})

    upper_bound = upper_bound.reset_index()
    upper_bound = upper_bound.rename(columns={'x': 'Lower Bound'})

    merged_df = pd.merge(lower_bound, upper_bound, on="index")

    merged_df['index'] = merged_df['index'].div(200)

    x2 = merged_df["Lower Bound"]
    y2 = merged_df["index"]

    plt.plot(x2, y2, label="Upper Bound of P_box", color='black', linestyle='solid',linewidth=2)

    x = merged_df["Upper Bound"]
    y = merged_df["index"]

    plt.plot(x, y, label="Lower Bound of P_box", color='black', linestyle='dashed',linewidth=2)

    if xlabel != None:
        plt.xlabel(xlabel,size=size)

    if ylabel != None:
        plt.ylabel(ylabel,size=size)

    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)

    if xlim != None:
        plt.xlim(xlim)

    if show_legend:
        plt.legend(fontsize=18)

    if show:
        plt.show()
