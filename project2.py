from sklearn.datasets import fetch_20newsgroups
from util_modified import *
from global_data import *

'''
comp_tech_subclasses = ['comp.graphics', 
                        'comp.os.ms-windows.misc', 
                        'comp.sys.ibm.pc.hardware', 
                        'comp.sys.mac.hardware']
                        
rec_act_subclasses = ['rec.autos', 
                      'rec.motorcycles', 
                      'rec.sport.baseball', 
                      'rec.sport.hockey']
train_data = fetch_20newsgroups(subset='train', categories=comp_tech_subclasses+rec_act_subclasses, shuffle=True, random_state=42)
test_data = fetch_20newsgroups(subset='test', categories=comp_tech_subclasses+rec_act_subclasses, shuffle=True, random_state=42)
'''


from timeit import default_timer as timer

logging.info("Problem 1")
start = timer()

X_train_tfidf = None
if os.path.isfile("./train_tfidf.pkl"):
    logging.info("Loading tfidf vector.")
    X_train_tfidf = pickle.load(open("./train_tfidf.pkl", "rb"))
else:
    X_train_tfidf = doTFIDF(train_data.data, MIN_DF)
    pickle.dump(X_train_tfidf, open( "./train_tfidf.pkl", "wb"))

print("With min_df = %d , (training documents, terms extracted): " % MIN_DF, X_train_tfidf.shape)

duration = timer() - start
logging.debug("Computation Time in secs: %d" % duration)
logging.info("finished Problem 1")

logging.info("Problem 2")
start = timer()



duration = timer() - start
logging.debug("Computation Time in secs: %d" % duration)
logging.info("finished Problem 2")