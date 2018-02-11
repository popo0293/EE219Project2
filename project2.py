from util_modified import *
from global_data import *

from timeit import default_timer as timer

GET_DATA_FROM_FILES = True

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

report_stats(train_label, km_pred.labels_, CAT)

duration = timer() - start
logging.debug("Computation Time in secs: %d" % duration)
logging.info("finished Problem 2")


logging.info("Problem 3")
start = timer()

ratio = None
svd = None
R_MAX = 1000
if GET_DATA_FROM_FILES and os.path.isfile("./var_ratio.pkl") and os.path.isfile("./svd1000.pkl"):
    logging.info("Loading explained variance ratio.")
    ratio = pickle.load(open("./var_ratio.pkl", "rb"))
    svd = pickle.load(open("./svd1000.pkl", "rb"))
else:
    svd = TruncatedSVD(n_components=R_MAX, n_iter=7, random_state=42)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    ratio = svd.explained_variance_ratio_.cumsum()
    pickle.dump(ratio, open("./var_ratio.pkl", "wb"))
    pickle.dump(svd, open("./svd1000.pkl", "wb"))

plt.plot(range(R_MAX), ratio, 'r', lw=3, label='Cumulative explained variance ratio')
plt.ylabel('Cumulative explained variance ratio')
plt.xlabel('r')
plt.show()

'''
r = [1, 2, 3, 5, 10, 20, 50, 100, 300]
y = []
for i in r:
    svd = TruncatedSVD(n_components=i)
    normalizer = Normalizer(copy=False)
    pipeline = make_pipeline(svd, normalizer)
    X_train_lsi = pipeline.fit_transform(X_train_tfidf)
    kmean = cluster_kmean(X_train_lsi, 2)
    result = report_stats(train_label, kmean.labels_, CAT, display=False)
    y.append(result)

y_transpose = np.array(y).T.tolist()

plt.plot(r, y_transpose[0], 'r', lw=6, label='homogenity')
plt.plot(r, y_transpose[1], 'y', lw=4, label='completeness')
plt.plot(r, y_transpose[4], 'k', lw=2, label='normalized mutual score')
plt.plot(r, y_transpose[3], label='rand score')
plt.ylabel('Singular Value', fontsize = 20)
plt.xlabel('Index', fontsize = 20)
plt.title('Top 1000 singular values', fontsize = 20)
plt.xscale('log')
plt.show()
'''
'''
duration = timer() - start
logging.debug("Computation Time in secs: %d" % duration)
logging.info("finished Problem 3")
'''