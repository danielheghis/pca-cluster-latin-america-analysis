import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hclust


_COLORS = ['y', 'r', 'b', 'g', 'c', 'm', 'sienna', 'coral',
           'darkblue', 'lime', 'grey',
           'tomato', 'indigo', 'teal', 'orange', 'darkgreen']


def plot_clusters(x, y, g, groups, labels=None, title="Plot clusters"):
    '''
    x, y - expect numpy.ndarray
    g - number of groups
    '''
    g_ = np.array(g)
    f = plt.figure(figsize=(12, 7))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=14, color='k')
    noOfGroups = len(_COLORS)
    for v in groups:
        x_ = x[g_ == v]
        y_ = y[g_ == v]
        k = int(v[1:])
        if len(x_) == 1:  # Cluster singleton
            ax.scatter(x_, y_, color='k', label=v)
        else:
            ax.scatter(x_, y_, color=_COLORS[k % noOfGroups], label=v)
    ax.legend()
    if labels is not None:
        for i in range(len(labels)):
            ax.text(x[i], y[i], labels[i])


def histograms(x, g, var):
    groups = set(g)
    g_ = np.array(g)
    m = len(groups)
    l = np.trunc(np.sqrt(m))
    if l * l != m:
        l += 1
    c = m // l
    if c * l != m:
        c += 1
    axes = []
    f = plt.figure(figsize=(12, 7))
    for i in range(1, m + 1):
        ax = f.add_subplot(int(l), int(c), int(i))
        axes.append(ax)
        ax.set_xlabel(var, fontsize=12, color='k')
    for v, ax in zip(groups, axes):
        y = x[g_ == v]
        ax.hist(y, bins=10, label=v, rwidth=0.9,
                range=(min(x), max(x)))
        ax.legend()


def dendrogram(h, labels=None, title='Hierarchical classification',
               threshold=None, colors=None):
    '''
    h - expect a numpy.ndarray hierarchy
    '''

    f = plt.figure(figsize=(12, 7))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=14, color='k')
    if colors is None:
        hclust.dendrogram(h, labels=labels, leaf_rotation=30,
                          ax=ax, color_threshold=threshold)
    else:
        hclust.dendrogram(h, labels=labels, leaf_rotation=30, ax=ax,
                          link_color_func=lambda k: colors[k])
    # if threshold is not None:
        # plt.axhline(y=threshold, color='r')


def show():
    plt.show()


