from .utils import getFileStrings
from .utils import getDTM
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def getCosSimRankResume(jobReqDir, resumeDir):
    docDtls = getFileStrings(jobReqDir, resumeDir)

    # for now assume one job to many candidate resumes. The job is the first
    # element
    docNames = docDtls[0]
    docStrs = docDtls[1]

    # get TF-IDF document term matrix that contains unigram to pentagrams
    dtm = getDTM(docNames, docStrs, True, 5)

    # apply SVD
    P, D, Q = np.linalg.svd(dtm, full_matrices=False)

    # account for number of singular values that explain at least 90% of the
    # variance
    var = (D ** 2) / sum(D ** 2)
    indx = np.asscalar(np.argwhere(np.cumsum(var) >= 0.90)[0])

    # reduce dimensions of document term matrix
    svd = TruncatedSVD(n_components=indx)
    svd.fit(dtm)
    # after transformation, result has number of rows = number of document and
    # columns = indx
    result = svd.transform(dtm)

    # compute cosine similarity between first row and remaining rows
    cosScores = [np.asscalar(cosine_similarity([result[0]], [result[x]]))
                 for x in range(1, result.shape[0])]

    # create series with cosine similarity scores along with resume doc names
    cosScores = pd.Series(cosScores, index=dtm.index.values[1:])

    # return sorted series in descending order of cosine similarity score
    cosScores.sort_values(axis=0, ascending=False, inplace=True)

    return cosScores