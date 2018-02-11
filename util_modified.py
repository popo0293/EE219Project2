
import string
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF, TruncatedSVD
import nltk
from nltk.tokenize import RegexpTokenizer

from sklearn.metrics import *
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.cluster import homogeneity_score,completeness_score, adjusted_rand_score, adjusted_mutual_info_score
from scipy.sparse.linalg import svds

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


def cluster_kmean(data):
    km = KMeans(n_clusters=2, max_iter=100, verbose=False, random_state=42).fit(data)
    return km


def test_stem_count_vectorize():
    test_string = ["Hello, Google. But I can't answer this call go going goes bowl bowls bowled!"]
    vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)
    X = vectorizer.fit_transform(test_string)
    feature_name = vectorizer.get_feature_names()
    print(feature_name)
    print(X.toarray())


def report_stats(label, predict, classes):
    n = len(classes)
    cmatrix = confusion_matrix(label, predict)
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

    print("Homogeneity: %0.3f" % homogeneity)
    print("Completeness: %0.3f" % completeness)
    print("V-measure: %0.3f" % v_measure)
    print("Adjusted Rand-Index: %.3f" % adjusted_Rand_Index)
    print("Adjusted Mutual Info Score: %0.3f" % adjusted_Mutual_Info_Score)
    return homogeneity, completeness, v_measure, adjusted_Rand_Index, adjusted_Mutual_Info_Score


def analyze(label, prob, predict, classes, n):
    if n <= 2:
        fpr, tpr, thresholds = roc_curve(label, prob)
        roc_auc = auc(fpr,tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='lightsteelblue',
                 lw=2, label='AUC (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='deeppink', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    cmatrix = confusion_matrix(label, predict)
    plt.imshow(cmatrix, interpolation='nearest', cmap=plt.cm.BuGn)
    plt.title("Confusion Matrix")
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=25)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cmatrix.max() / 2.
    for i, j in itertools.product(range(n), range(n)):
        plt.text(j, i, format(cmatrix[i, j], fmt),
                horizontalalignment="center",
                color="white" if cmatrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

    print("accuracy: ", accuracy_score(label, predict))
    if n <= 2:
        print("recall: ", recall_score(label, predict))
        print("precision: ", precision_score(label, predict))
    else:
        print("recall: ", recall_score(label, predict, average='weighted'))
        print("precision: ", precision_score(label, predict, average='weighted'))
    return


