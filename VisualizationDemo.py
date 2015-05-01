import readnarratives
import visualizeresults
import scoresentencesdata
import numpy
import random
import scipy
import math

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def mostInformativeFeatures(classifier, vectorizer, 
	categories,number_of_features):
	"""This takes in a classfier and spits out the n most informative features
	for each category"""
	#Get all the names of the reatures
	feature_names = numpy.asarray(vectorizer.get_feature_names())
	for i, category in enumerate(categories):
		#For every cattegory et the top n features
		#To do this, we sort the reatures in ascending order and get 
		#Last n
		top = numpy.argsort(classifier.coef_[i])[-number_of_features:]

		print("%s: %s" % (category, ", ".join(feature_names[top])))

def mostInformativeFeaturesRegression (classifier, vectorizer,
	number_of_features):
	feature_names = numpy.asarray(vectorizer.get_feature_names())
	sorted_coefs = numpy.argsort(classifier.coef_)
	top_pos = sorted_coefs[-number_of_features:]
	top_neg = sorted_coefs[0:number_of_features:]
	print("Top Positive: %s" % ", ".join(feature_names[top_pos]))
	print("Top Negative: %s" % ", ".join(feature_names[top_neg])) 

def splitTestTrain(X,y, train_size = 0.66, random_state = 42):
	#Making my own so I can split text!
	random.seed(random_state)
	shuffle_X = X
	shuffle_y = y
	random.shuffle(shuffle_X)
	random.shuffle(shuffle_y)

	splitindex = int(math.ceil(len(y)*train_size))

	X_train = shuffle_X[:splitindex]
	X_test = shuffle_X[splitindex:]

	y_train = shuffle_y[:splitindex]
	y_test = shuffle_y[splitindex:]

	return X_train, X_test, y_train, y_test


def main():
	numpy.set_printoptions(threshold=numpy.nan)

	#Read Files
	pickle_name = 'NarrativePickleAgency_test'
	responses = readnarratives.readPickle(pickle_name)

	sentences = scoresentencesdata.getSentenceData(responses)

	texts = [sentence[0] for sentence in sentences]
	scores_actual = [sentence[1] for sentence in sentences]

	scores_dict = {}
	scores_dict[0.0] = -1
	scores_dict[1.0] = -1
	scores_dict[1.5] = 0
	scores_dict[2.0] = 1
	scores_dict[3.0] = 1

	scores_no_dec = [scores_dict[score] for score in scores_actual]

	#Split test train
	X_train, X_test, y_train, y_test = splitTestTrain(
		texts, scores_no_dec, 0.66, 42)

	tfidf_vect = TfidfVectorizer(token_pattern = r'\w*', 
		max_features = 450)

	X_train_tfidf = tfidf_vect.fit_transform(X_train)
	X_test_tfidf = tfidf_vect.fit_transform(X_test)

	

	#Logistic Regresssion
	#print('\n Logistic Regression \n')
	logistic_model = LogisticRegression(penalty = 'l2',
		tol = 0.00001, C=0.001, intercept_scaling=10000)
	logistic_model.fit(X_train_tfidf,y_train)
	
	predicted_logistic = logistic_model.predict(X_test_tfidf)


	print y_test
	print predicted_logistic


	# mostInformativeFeaturesRegression(logistic_model, tfidf_vect,15)

	mostInformativeFeatures(logistic_model, tfidf_vect,[-1,1,0], 15)
	#print(logistic_model.score(X_test_tfidf,y_test))
	print('Accuracy (r squared): %f' % logistic_model.score(
		X_test_tfidf,y_test))

	raw_input("Press Enter to continue...")

	to_print = range(len(y_test))

	visualizeresults.visualizeWeightsList(to_print, X_test,X_test_tfidf,
		predicted_logistic, y_test, tfidf_vect,logistic_model, True)

	visualizeresults.printKey()

if __name__ == '__main__':
	main()