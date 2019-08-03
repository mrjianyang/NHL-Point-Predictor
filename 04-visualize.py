import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import sys
import seaborn as sns
from scipy import stats

sns.set(style="white", color_codes=True)
def heat_map(df, x, y, rows, cols, index):  # rows, cols, index
    plt.subplot(rows,cols,index)
    plt.hist2d(df[x], df[y], cmap=plt.cm.Reds)
    plt.xlabel(x)
    plt.ylabel(y)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')

def create_heatmaps(df):
    heat_map(df, 'pts y/n l_3', 'points', 3, 2, 1)
    heat_map(df, 'pts y/n l_7', 'points', 3, 2, 2)
    heat_map(df, 'games l_21', 'points', 3, 2, 3)
    heat_map(df, 'games l_7', 'points', 3, 2, 4)
    heat_map(df, 'shots', 'points', 3, 2, 5)
    plt.savefig(y+".jpg", ppi = 1000, quality = 90)
    plt.show()

def scat_plot(df, x, y, rows, cols, index, fig):
    plt.subplot(rows,cols,index)
    # plt.title(pos + " " + x + " vs " + y)
    plt.plot(df[x], df[y], 'b.', alpha=0.5)
    plt.xlabel(x)
    plt.ylabel(y)
    # plt.legend("Correlation: " + str(df[x].corr(df[y])) )
    #print(x, df[x].corr(df[y]))

def histogram(df, x, rows, cols, index):
    plt.subplot(rows,cols,index)
    plt.hist(df[x])
    plt.xlabel(x)
    plt.ylabel('counts')

def create_hist(df_F):
    # forwards
    histogram(df_F, "timeOnIce",4, 2, 1)
    histogram(df_F, "shots", 4, 2, 2)
    histogram(df_F, "pts y/n l_7", 4, 2, 3)
    histogram(df_F, "pts y/n l_3", 4, 2, 4)
    histogram(df_F, "powerPlayTimeOnIce", 4, 2, 5)
    histogram(df_F, "ppg", 4, 2, 6)
    histogram(df_F, "games l_21", 4, 2, 7)
    histogram(df_F, "games l_7",  4, 2, 8)
    plt.savefig('hist_F.png', ppi = 1000)
    plt.show()

def create_scat_plots(df):
    # forwards
    scat_plot(df, "timeOnIce", 'points', 2, 2, 1, "A")
    # scat_plot(df_F, "shots",'points', "F", 4, 2, 2, "B")
    # scat_plot(df_F, "pts y/n l_7", 'points', "F", 4, 2, 3, "C")
    # scat_plot(df_F, "pts y/n l_3",'points', "F", 4, 2, 4, "D")
    scat_plot(df, "powerPlayTimeOnIce",'points',  2, 2, 2, "B")
    scat_plot(df, "ppg", 'points', 2, 2, 3, "C")
    # scat_plot(df_F, "games l_21",'points', "F",  4, 2, 7, "G")
    # scat_plot(df_F, "games l_7",'points', "F",  4, 2, 8, "H")
    # plt.tight_layout
    plt.savefig('scatter_F.png', ppi = 1000)
    plt.show()

def main(df):
    # scatterplots
    create_scat_plots(df)

    # histograms
    create_hist(df)

    # heatmaps
    create_heatmaps(df)

if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])
    main(df)
