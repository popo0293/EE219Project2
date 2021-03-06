
import string
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF, TruncatedSVD, PCA
import nltk
from nltk.tokenize import RegexpTokenizer

from sklearn.metrics import *
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.cluster import *

from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

'''
try:
    nltk.download("stopwords")  # if the host does not have the package
except (RuntimeError):
    pass
'''

# globals
MIN_DF = 3


class SparseToDenseArray(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, *_):
        if hasattr(X, 'toarray'):
            return X.toarray()
        return X

    def fit(self, *_):
        return self


tfidf_transformer = TfidfTransformer(smooth_idf=False)


def doTFIDF(data, mindf):
    vectorizer = CountVectorizer(min_df=mindf, stop_words=ENGLISH_STOP_WORDS)
    m = vectorizer.fit_transform(data)
    m_train_tfidf = tfidf_transformer.fit_transform(m)
    return m_train_tfidf


def cluster_kmean(data, n):
    km = KMeans(n_clusters=n, n_init=20, max_iter=100, verbose=False).fit(data)
    pred = km.predict(data)
    return pred


def report_stats(label, predict, classes, display=True, msg=None):
    n = len(classes)
    cmatrix = contingency_matrix(label, predict)
    if display:
        plt.imshow(cmatrix, interpolation='nearest', cmap=plt.cm.BuGn)
        plt.title("Contingency Table")
        tick_marks = np.arange(n)
        className = []
        for i in range(n):
            className.append(str(i))
        plt.xticks(tick_marks, className)
        plt.yticks(tick_marks, classes)
        fmt = 'd'
        thresh = cmatrix.max() / 2.
        for i, j in itertools.product(range(n), range(n)):
            plt.text(j, i, format(cmatrix[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cmatrix[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('Ground Truth Label')
        plt.xlabel('Cluster Label')
        plt.show()

    homogeneity = homogeneity_score(label, predict)
    completeness = completeness_score(label, predict)
    v_measure = v_measure_score(label, predict)
    adjusted_Rand_Index = adjusted_rand_score(label, predict)
    adjusted_Mutual_Info_Score = adjusted_mutual_info_score(label, predict)

    if isinstance(msg, str):
        print(msg)
    print("Homogeneity: %0.3f" % homogeneity)
    print("Completeness: %0.3f" % completeness)
    print("V-measure: %0.3f" % v_measure)
    print("Adjusted Rand-Index: %.3f" % adjusted_Rand_Index)
    print("Adjusted Mutual Info Score: %0.3f" % adjusted_Mutual_Info_Score)

    return [cmatrix, [homogeneity, completeness, v_measure, adjusted_Rand_Index, adjusted_Mutual_Info_Score]]



