import pandas as pd
import numpy as np
from . import utilsHCA as utils
from . import graphicsHCA as graphics
import scipy.cluster.hierarchy as hclust
import scipy.spatial.distance as hdist
import sklearn.decomposition as dec
import matplotlib as mpl
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


def AnalizaClusteringuluiIerarhic():
    try:

        fileName = 'dataIN/Date_AmericaLatina.csv'

        # set warning only when opened more than 50 figures
        mpl.rcParams['figure.max_open_warning'] = 50
        print(mpl.rcParams['figure.max_open_warning'])

        # List of options for the runtime,
        # keep in the list only the desired options
        drawing_options = ['Partition plot in main axes',
                           # 'Plot histograms',
                           'Variable grouping']
        discriminant_axes = (drawing_options.
                             __contains__('Partition plot in main axes'))
        histograms = (drawing_options.
                      __contains__('Plot histograms'))
        variable_grouping = (drawing_options.
                             __contains__('Variable grouping'))

        table = pd.read_csv(fileName, na_values='')
        utils.replace_na_df(table)

        labelVars = list(table)

        # Specify the column indices based on your dataset
        columnIndex = 'Country'
        columnLabel = 'Country'
        print(table[columnLabel].values)

        table.index = [str(v) for v in table[columnIndex]]
        print(table.index)
        table.index.name = columnIndex
        print(table.index.name)

        #display the dataframe
        print(table)

        # Extract necessary data for clustering
        obs = table[columnIndex].values
        print(obs)
        vars = labelVars[1:]
        X = table[vars].values
        Xstd = utils.standardise(X)

        # print(Xstd)

        # Creare ierarhie instante
        methods = list(hclust._LINKAGE_METHODS)
        metrics = hdist._METRICS_NAMES
        print('Methods: ', methods)
        print('Metrics: ', metrics)

        # method = methods[1]  # complete
        method = methods[5]  # ward
        # method = methods[6]  # weighted
        # method = methods[2]  #average
        distance = metrics[3]  # citiblock

        if (method == 'ward' or method == 'centroid' or
                method == 'median' or method == 'weighted'):
            distance = 'euclidean'
        else:
            distance = metrics[3]

        h = hclust.linkage(Xstd, method=method, metric=distance)
        # Maximum stability partition identification
        m = np.shape(h)[0]  # Maximum number of junctions
        # The index of the largest difference between two consecutive
        # junction distances is subtracted from m (maximum number of junctions)
        # to determine the number of clusters in the most stable partition
        k = m - np.argmax(h[1:m, 2] - h[:(m - 1), 2])
        # Identification of clusters in the maximum stability partition
        g_max, codes = utils.cluster_distribution(h, k)
        # cluster display
        utils.cluster_display(g_max, table.index, columnIndex,
                              'HCA/dataOUT/P_max.csv')
        # save cluster distribution in CSV file
        utils.cluster_save(g=g_max, row_labels=table.index.values,
                               col_label='Cluster',
                           file_name='OptimalPartition.csv')

        t_1, j_1, m_1 = utils.threshold(h)
        print('threshold=', t_1, 'junction with the maximum difference=', j_1,
              'no. of junctions=', m_1)

        # Determination of cluster colors
        color_clusters = utils.color_clusters(h, k, codes)
        graphics.dendrogram(h, labels=table[columnLabel].values,
                    title='Observations clustering. Partition of maximum stability. '
                          'Method: ' +
                    method + 'Metric: ' + distance,
                            colors=color_clusters, threshold=t_1)
        if discriminant_axes:
            model_pca = dec.PCA(n_components=2)
            z = model_pca.fit_transform(Xstd)
            groups = list(set(g_max))

            # Use the 'Country' column for labeling
            graphics.plot_clusters(z[:, 0], z[:, 1], g_max, groups, labels=table['Country'].values,
                                   title='Optimal partition')

        if histograms:
            for v in vars:
                graphics.histograms(table[v].values, g_max, var=v)

        # Variable hierarchy
        if variable_grouping:
            # method_v = methods[5]  # average
            method_v = 'average'
            distance_v = 'correlation'

            h2 = hclust.linkage(X.transpose(), method=method_v, metric=distance_v)
            t_2, j_2, m_2 = utils.threshold(h2)
            print('threshold=', t_2, 'junction with the maximum difference=', j_2,
                  'no. of junctions=', m_2)
            graphics.dendrogram(h2, labels=vars,
                            title="Variable clustering. Method: " + method_v +
                            " Metric: " + distance_v, threshold=t_2)

        n = np.shape(X)[0]
        # Prepare a selection list for the partitions,
        # starting from the partition with two clusters
        list_selections = [str(i) + ' clusters' for i in range(2, n - 1)]
        partitions = list_selections[0:5]

        # Create DataFrame with the maximum stability partition
        # and the selected partitions
        t_partitions = pd.DataFrame(index=table.index)
        t_partitions['P_max'] = g_max
        for v in partitions:
            k = list_selections.index(v) + 2  # Desired number of clusters
            g, codes = utils.cluster_distribution(h, k)

            # Save partition
            utils.cluster_display(g, table.index, columnIndex,
                                  './HCA/dataOUT/P_' + str(k) + '.csv')
            utils.cluster_save(g=g, row_labels=table.index.values,
                               col_label='Cluster', file_name='Partition_' + str(k) + '.csv')

            # Use the 'Country' column for labeling in partition plots
            color_clusters = utils.color_clusters(h, k, codes)
            graphics.dendrogram(h, table[columnLabel],
                                title='Partition with ' + v,
                                colors=color_clusters, threshold=t_1)
            t_partitions['P_' + v] = g
            if discriminant_axes:
                model_pca = dec.PCA(n_components=2)
                z = model_pca.fit_transform(Xstd)
                groups = list(set(g))

                # Use the 'Country' column for labeling in cluster plot
                graphics.plot_clusters(z[:, 0], z[:, 1], g, groups, labels=table['Country'].values,
                                       title='Partition with ' + v)
            # if histograms:
            #     for v in vars:
            #         graphics.histograms(table[v].values, g, var=v)

        t_partitions.to_csv('./HCA/dataOUT/Partitions.csv')

        # Metoda Silhouette

        # Set de date prelucrat
        df = table.drop(table.columns[0], axis=1)

        for k in range(2, 6):
            # Fit KMeans clustering model
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(df)

            # Calculam silhouette score
            silhouette_avg = silhouette_score(df, cluster_labels)
            print(f"For n_clusters = {k}, the average silhouette score is : {silhouette_avg}")

        graphics.show()


    except Exception as ex:
        print("Error!", ex.with_traceback(), sep="\n")






