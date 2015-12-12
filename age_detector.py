# -*- coding: utf-8 -*-
__author__ = 'Dmitriy Ovchinnikov'

from sklearn.preprocessing import LabelEncoder
from time import time
import json
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.preprocessing import Normalizer
#from sklearn.decomposition import TruncatedSVD
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
import re
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
#from sklearn.svm import SVC
from sklearn import metrics
import multiprocessing

class AgeDetector:

    def __init__(self):
        """
        it is constructor
        """
        self.le = LabelEncoder()
        self.clf = None
        
    def train(self, instances, labels, research = False):
        """
        Train method
        """
        l = self.le.fit_transform(labels)
        texts = []
        #print("Preparing...")
        #t0 = time()
        for user in instances:
        	result = ""
        	for replica in user:
        		result += (" " + replica)
       		#print(result)
        	texts.append(result)
        #print("Preparing finished %0.1fs" % (time() - t0))

        if (research):
        	txt_clf = Pipeline([
        		('vect', CountVectorizer(ngram_range=(1, 7), analyzer='char_wb')),
        		('tf_idf', TfidfTransformer(norm='l2')),
        		#('norm', Normalizer()),
        		('clf', SVC())])
        	param = {
        		#'vect__analyzer' : ('char_wb', 'word'),
        		#'vect__ngram_range' : ((1,7), (1, 8), (1, 10)),
        		'vect__max_df' : (0.3, 0.5, 1.0),
        		#'tf_idf__norm' : ('l1', 'l2'),
				#'clf__alpha' : (0.01, 0.03, 0.05)
        		#'clf__n_iter' : (10, 50, 70),
        		'clf__C' : (0.1, 0.5, 0.7, 1.0)
        	}
        	t_gs = time()
        	cr_val = cross_validation.StratifiedKFold(l, n_folds=15)
        	gs_clf = GridSearchCV(txt_clf, param, n_jobs=-1, cv=cr_val, verbose=1)
        	gs_clf.fit(texts, l)
        	print("GridSearch finished! %0.3fs" % (time()-t_gs))
        	best_parametres, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
        	for param_name in sorted(param.keys()):
        		print("%s: %r" % (param_name, best_parametres[param_name]))
        	print("Score: %0.4f" % score)
        	#f = open('./best_score.txt', 'w')
        	#f.write(score)
        	#f.close()
        	self.clf = gs_clf.best_estimator_
        else:
        	self.clf = Pipeline([
        		('prep', Preparator(n_jobs=-1)),
        		('vect', CountVectorizer(analyzer='char_wb', ngram_range=(1, 8), max_df=0.5)),
        		#('vect', HashingVectorizer(n_features=(2 ** 20), 
        		#	analyzer='char_wb', 
        		#	non_negative=True, 
        		#	norm=None, 
        		#	ngram_range=(1, 7))),
                ('tf_idf', TfidfTransformer(norm='l2', sublinear_tf=True)),
        		#('lsa', TruncatedSVD(n_components=100)),
                #('norm', Normalizer(copy=False)),
                #('clf', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2)))])
                #('clf', SVC(C=0.1, verbose=True))])
        		('clf', OneVsRestClassifier(PassiveAggressiveClassifier(C=0.9, n_iter=70, n_jobs=-1), n_jobs=-1))])
        		#('clf', MultinomialNB(alpha=0.03))])
				
        	print("Training started!")
        	t = time()
        	self.clf.fit(texts, l)
        	print("Training succesfully finished. %0.1fs" % (time() - t))

    def classify(self, instances):
        """
        Classify method 
        """
        texts = []
        for user in instances:
        	res = ""
        	for rep in user:
        		res += (" " + rep)
        	texts.append(res)
        return self.le.inverse_transform(self.clf.predict(texts))

class Preparator(object):
	def __init__(self, n_jobs=2):
		try:
			cpus = multiprocessing.cpu_count()
		except NotImplementedError:
			cpus = 1
		if ((n_jobs > cpus) or (n_jobs==-1)):
			self.n_jobs = cpus
		else:
			self.n_jobs = n_jobs
		self.pool = multiprocessing.Pool(processes=self.n_jobs)
	def prepare(self, text):
		text = text.lower()
		text = re.sub(r"[\)]+", ')', text)
		text = re.sub(r"[\(]+", '(', text)
		text = re.sub(r"[ ]([0-9])[ ]", ' ', text)
		text = re.sub(r"(?:http.+)([ ]|$)+", ' ', text)
		return text
	def transform(self, X):
		result = self.pool.map(self.prepare, X)
		print("Preparing finished...")
		return result
	def fit(self, X, y=None):
		print("Preparing...")
		return self
	def __getstate__(self):
		self_dict = self.__dict__.copy()
		del self_dict['pool']
		return self_dict
	def __setstate__(self, state):
		self.__dict__.update(state)

if __name__ == '__main__':
	print("Loading corpus...")
	t = time()
	instances = json.load(open('./Train.txt.json'))
	labels = json.load(open('./Train.lab.json'))
	print("Corpus loaded. %0.1fs" % (time() - t))
	detector = AgeDetector()
	detector.train(instances[:1000], labels[:1000], research=False)
	predicted = detector.classify(instances[300:800])
	expected = labels[300:800]
	print(metrics.classification_report(expected, predicted))
