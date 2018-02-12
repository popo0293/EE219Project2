from util_modified import *
from global_data import *

from timeit import default_timer as timer

GET_DATA_FROM_FILES = True
DETAILS = False

logging.info("Problem 1")
start = timer()

X_train_tfidf = None
if GET_DATA_FROM_FILES and os.path.isfile("./train_tfidf.pkl"):
    logging.info("Loading tfidf vector.")
    X_train_tfidf = pickle.load(open("./train_tfidf.pkl", "rb"))
else:
    X_train_tfidf = doTFIDF(train_data.data, MIN_DF)
    pickle.dump(X_train_tfidf, open("./train_tfidf.pkl", "wb"))

print("With min_df = %d , (training documents, terms extracted): " % MIN_DF, X_train_tfidf.shape)

duration = timer() - start
logging.debug("Computation Time in secs: %d" % duration)
logging.info("finished Problem 1")

logging.info("Problem 2")
start = timer()

km_pred = None
if GET_DATA_FROM_FILES and os.path.isfile("./kmean.pkl"):
    logging.info("Loading predicted kmean.")
    km_pred = pickle.load(open("./kmean.pkl", "rb"))
else:
    km_pred = cluster_kmean(X_train_tfidf, 2)
    pickle.dump(km_pred, open("./kmean.pkl", "wb"))

report_stats(train_label, km_pred, CAT)

duration = timer() - start
logging.debug("Computation Time in secs: %d" % duration)
logging.info("finished Problem 2")


logging.info("Problem 3")
start = timer()

R_MAX = 1000
ratio = None
if GET_DATA_FROM_FILES and os.path.isfile("./ratio.pkl"):
    logging.info("Loading ratio.")
    ratio = pickle.load(open("./ratio.pkl", "rb"))
else:
    svd = TruncatedSVD(n_components=R_MAX, n_iter=7, random_state=42)
    svd.fit_transform(X_train_tfidf)
    ratio = svd.explained_variance_ratio_.cumsum()
    pickle.dump(ratio, open("./ratio.pkl", "wb"), True)

plt.plot(range(R_MAX), ratio, 'r', lw=3, label='Cumulative explained variance ratio')
plt.ylabel('Cumulative explained variance ratio')
plt.xlabel('r')
plt.show()

r = [1, 2, 3, 5, 10, 20, 50, 100, 300]

# LSI
y = None
cmatrix = None
if GET_DATA_FROM_FILES and not DETAILS \
        and os.path.isfile("./y_lsi.pkl") \
        and os.path.isfile("./cmatrix_lsi.pkl"):
    logging.info("Loading y and cmatrix for LSI.")
    y = pickle.load(open("./y_lsi.pkl", "rb"))
    cmatrix = pickle.load(open("./cmatrix_lsi.pkl", "rb"))
else:
    y = []
    cmatrix = []
    for i in r:
        svd = TruncatedSVD(n_components=i, random_state=None)
        # normalizer = Normalizer(copy=False)
        # pipeline = make_pipeline(svd, normalizer)
        # X_train_lsi = pipeline.fit_transform(X_train_tfidf)
        X_train_lsi = svd.fit_transform(X_train_tfidf)
        kmean = cluster_kmean(X_train_lsi, 2)
        msg = 'With r = %d' % i + " Using LSI"
        result = report_stats(train_label, kmean, CAT, display=False, msg=msg)
        print("-  "*10)
        print("The contingency matrix is: ")
        cmatrix.append(result[0])
        print(result[0])
        y.append(result[1])
        print("-"*30)

    pickle.dump(y, open("./y_lsi.pkl", "wb"), True)
    pickle.dump(cmatrix, open("./cmatrix_lsi.pkl", "wb"), True)

y_transpose = np.array(y).T.tolist()

r_len = len(r)
l1, = plt.plot(range(r_len), y_transpose[0], 'r', lw=4, label='homogenity')
l2, = plt.plot(range(r_len), y_transpose[1], 'g', lw=2, label='completeness')
l3, = plt.plot(range(r_len), y_transpose[2], 'b', lw=2, label='completeness')
l4, = plt.plot(range(r_len), y_transpose[3], 'm', lw=2, label='rand index')
l5, = plt.plot(range(r_len), y_transpose[4], 'k', lw=2, label='adjusted mutual information')
tick_marks = np.arange(r_len)
labels = [str(a) for a in r]
plt.xticks(tick_marks, labels)
plt.legend(handles=[l1, l2, l3, l4, l5])
plt.xlabel('r')
plt.show()

best_r = [np.argmax(y_transpose[i]) for i in range(5)]
print("*"*60)
bi = np.bincount(best_r).argmax() # best_r_index
print("The best R value for TruncatedSVD is %d" % r[bi])
print("The contingency matrix is: ")
print(cmatrix[bi])
print("Homogeneity: %0.3f" % y_transpose[0][bi])
print("Completeness: %0.3f" % y_transpose[1][bi])
print("V-measure: %0.3f" % y_transpose[2][bi])
print("Adjusted Rand-Index: %.3f" % y_transpose[3][bi])
print("Adjusted Mutual Info Score: %0.3f" % y_transpose[4][bi])
print("*"*60)


# NMF
if GET_DATA_FROM_FILES and not DETAILS \
        and os.path.isfile("./y_nmf.pkl") \
        and os.path.isfile("./cmatrix_nmf.pkl"):
    logging.info("Loading y and cmatrix for NMF.")
    y = pickle.load(open("./y_nmf.pkl", "rb"))
    cmatrix = pickle.load(open("./cmatrix_nmf.pkl", "rb"))
else:
    y = []
    cmatrix = []
    for i in r:
        svd = NMF(n_components=i)
        # normalizer = Normalizer(copy=False)
        # pipeline = make_pipeline(svd, normalizer)
        # X_train_lsi = pipeline.fit_transform(X_train_tfidf)
        X_train_nmf = svd.fit_transform(X_train_tfidf)
        kmean = cluster_kmean(X_train_nmf, 2)
        msg = 'With r = %d' % i + " Using NMF"
        result = report_stats(train_label, kmean, CAT, display=False, msg=msg)
        print("-  "*10)
        print("The contingency matrix is: ")
        cmatrix.append(result[0])
        print(result[0])
        y.append(result[1])
        print("-"*30)
    pickle.dump(y, open("./y_nmf.pkl", "wb"), True)
    pickle.dump(cmatrix, open("./cmatrix_nmf.pkl", "wb"), True)

y_transpose = np.array(y).T.tolist()

r_len = len(r)
l1, = plt.plot(range(r_len), y_transpose[0], 'r', lw=4, label='homogenity')
l2, = plt.plot(range(r_len), y_transpose[1], 'g', lw=2, label='completeness')
l3, = plt.plot(range(r_len), y_transpose[2], 'b', lw=2, label='completeness')
l4, = plt.plot(range(r_len), y_transpose[3], 'm', lw=2, label='rand index')
l5, = plt.plot(range(r_len), y_transpose[4], 'k', lw=2, label='adjusted mutual information')
tick_marks = np.arange(r_len)
labels = [str(a) for a in r]
plt.xticks(tick_marks, labels)
plt.legend(handles=[l1, l2, l3, l4, l5])
plt.xlabel('r')
plt.show()

best_r = [np.argmax(y_transpose[i]) for i in range(5)]
print("*"*60)
bi = np.bincount(best_r).argmax() # best_r_index
print("The best R value for NMF is %d" % r[bi])
print("The contingency matrix is: ")
print(cmatrix[bi])
print("Homogeneity: %0.3f" % y_transpose[0][bi])
print("Completeness: %0.3f" % y_transpose[1][bi])
print("V-measure: %0.3f" % y_transpose[2][bi])
print("Adjusted Rand-Index: %.3f" % y_transpose[3][bi])
print("Adjusted Mutual Info Score: %0.3f" % y_transpose[4][bi])
print("*"*60)

duration = timer() - start
logging.debug("Computation Time in secs: %d" % duration)
logging.info("finished Problem 3")
