from src.utils import getFileStrings
from src.utils import getDTM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skrebate import ReliefF
import pandas as pd
import numpy as np


def getClustRankResume(resumeDir):
    docDtls = getFileStrings(None, resumeDir)

    docNames = docDtls[0]
    docStrs = docDtls[1]

    # get document term matrix based on bag of words that contains only unigrams
    dtm = getDTM(docNames, docStrs, False, 1)

    # determine optimal k parameter
    k = 0
    previous_silh_avg = 0.0
    for n_clusters in range(2, dtm.shape[0]):
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(dtm)
        silhouette_avg = silhouette_score(dtm, cluster_labels)
        if silhouette_avg > previous_silh_avg:
            previous_silh_avg = silhouette_avg
            k = n_clusters

    print('Optimal k value determined = ' + str(k))

    # Final Kmeans for best_clusters. This returns array of cluster numbers
    kmeans = KMeans(n_clusters=k).fit_predict(dtm)

    # determine relieff weight of each unigram
    fs = ReliefF()
    fs.fit(dtm.values, kmeans)

    # multiply count of words with corresponding weight to get cluster based
    # ranking
    cbr = np.dot(dtm, fs.feature_importances_)

    # create series with relieff weight along with resume doc names
    return pd.Series(cbr, index=docNames)
