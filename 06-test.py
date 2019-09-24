import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
import seaborn as sns
import pickle
sns.set(style="white")

default_cmap = cm.get_cmap('rainbow')


def get_data(filename):  
    df = pd.read_csv(filename)
    df['avg'] = (df['ppg'] + df['powerPlayTimeOnIce']) / 2
    X = df[['timeOnIce', 'shots', 'powerPlayTimeOnIce', 'ppg']].values#, 'ppg', 'pts l_7', 'pts l_3', 'pp pts l_7', 'pp pts l_3'  
    y = df['points y/n'].values
    #print(df['points y/n'].value_counts())
    return X, y, df

def get_train_data(X, y):
    return train_test_split(X, y,test_size=0.25,random_state=0)

def get_decision_matrix(X_valid, y_valid, model):
    y_pred= model.predict(X_valid)
    cnf_matrix = metrics.confusion_matrix(y_valid, y_pred)
    print(cnf_matrix)

    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png", ppi=1000)
    print("Accuracy:",metrics.accuracy_score(y_valid, y_pred))
    print("Precision:",metrics.precision_score(y_valid, y_pred))
    print("Recall:",metrics.recall_score(y_valid, y_pred))
    plt.show()

def plot_decision(model, df, X, y=None, width=400, height=400, cmap=None):
    if cmap is None:
        cmap = default_cmap
    # assumes >= 2 features. Plots first as x axis; second as y.
    Q0 = df['powerPlayTimeOnIce']
    Q1 = df['ppg']
    x0 = X[:, 0] #column 0
    y0 = X[:, 1] #column 1
    x10 = X[: 2]
    y10 = X[: 3]

    xg = np.linspace(x0.min(), x0.max(), width)
    yg = np.linspace(y0.min(), y0.max(), height)
    Qg = np.linspace(Q0.min(), Q0.max(), height)
    Qy = np.linspace(Q1.min(), Q1.max(), height)

    xg1 = np.linspace(x0.min(), x0.max(), width)
    yg1 = np.linspace(y0.min(), y0.max(), height)

    xx, yy = np.meshgrid(xg, yg)
    qxx, qyy = np.meshgrid(Qg, Qy)

    X_grid = np.vstack([xx.ravel(), yy.ravel(), qxx.ravel(), qyy.ravel()]).T
    y_grid = model.predict(X_grid)
    plt.contourf(xx, yy, y_grid.reshape((height, width)), cmap=cmap) #plots the background
    if y is not None:
        plt.scatter(x0, y0, c=y, cmap=cmap, edgecolor='k')
    plt.ylabel("Shots")
    plt.xlabel("Time On Ice")
    plt.savefig('model.png', ppi = 1000)
    plt.show()

def main():
    # load data and model
    X, y, df = get_data(sys.argv[1])
    training_model = pickle.load(open(sys.argv[2], 'rb'))
    
    # split data
    X_train, X_valid, y_train, y_valid = get_train_data(X, y)

    # train/test
    print(training_model.score(X_train, y_train))
    print(training_model.score(X_valid, y_valid))
    plot_decision(training_model, df, X, y)

    get_decision_matrix(X_valid, y_valid, training_model)

if __name__ == '__main__':
    main()