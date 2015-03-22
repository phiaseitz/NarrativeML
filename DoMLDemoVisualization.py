import readnarratives
import visualizeresults
import numpy
import random
import scipy
import math
import logisticordinalregression
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier,Ridge,RidgeCV,LinearRegression
from sklearn.linear_model import RidgeCV

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
	pickle_name = 'NarrativePickleAgency'
	texts, scores = readnarratives.readPickle(pickle_name)


	#Split test train
	X_train, X_test, y_train, y_test = splitTestTrain(
		texts, scores, 0.66, 42)

	#count_vect = CountVectorizer(stop_words = "english")
	#count_vect = CountVectorizer(ngram_range=(1, 2),stop_words = "english")
	#count_vect = CountVectorizer(ngram_range=(1, 2))
	count_vect = CountVectorizer(token_pattern = r'\w*')
	#count_vect = CountVectorizer()

	X_train_count = count_vect.fit_transform(X_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)
	X_train_tfidf_a = tfidf_transformer.fit_transform(X_train_count).toarray()

	X_test_count = count_vect.transform(X_test)
	X_test_tfidf = tfidf_transformer.transform(X_test_count)
	X_test_tfidf_a = tfidf_transformer.transform(X_test_count).toarray()
	

	#Ridge Regression
	print('\nRidge Regression \n')
	ridge_model = Ridge(alpha = 10) #When only doing bag of words, 10 is good
	ridge_model.fit(X_train_tfidf,y_train)

	predicted_ridge = ridge_model.predict(X_test_tfidf)

	#print(ridge_model.alpha_)
	print y_test
	print predicted_ridge
	reliability_ridge = scipy.stats.pearsonr(predicted_ridge,y_test)

	#print(ridge_model.coef_)

	mostInformativeFeaturesRegression(ridge_model, count_vect,15)
	#print(ridge_model.score(X_test_tfidf,y_test))
	print('Accuracy (r squared): %f' % ridge_model.score(X_test_tfidf,y_test))

	raw_input("Press Enter to continue...")

	to_print = [0,1,8]

	visualizeresults.visualizeWeightsList(to_print, X_test,X_test_tfidf,
		y_test, count_vect,ridge_model)

	visualizeresults.printKey()

if __name__ == '__main__':
	main()