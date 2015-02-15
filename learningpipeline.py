import readnarratives
import numpy
import random
import scipy
import math

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import SGDClassifier

def mostInformativeFeatures(classifier, vectorizer, categories,number_of_features):
	feature_names = numpy.asarray(vectorizer.get_feature_names())
	for i, category in enumerate(categories):
		top = numpy.argsort(classifier.coef_[i])[-number_of_features:]
		print("%s: %s" % (category, " ".join(feature_names[top])))


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

	count_vect = CountVectorizer()
	X_train_count = count_vect.fit_transform(X_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

	X_test_count = count_vect.transform(X_test)
	X_test_tfidf = tfidf_transformer.transform(X_test_count)
	
	
	#Naive Bayes - I'm not sure but this might not work so well
		#always predicts 1...
	print ('Naive Bayes')
	naive_bayes_model = MultinomialNB().fit(X_train_tfidf, y_train)

	predicted_nb = naive_bayes_model.predict(X_test_tfidf)
	print('Accuracy: %f' % numpy.mean(predicted_nb == y_test))

	print y_test
	print predicted_nb

	mostInformativeFeatures(naive_bayes_model, count_vect,[0,1,2,3],10)

	#Support vector
	print('Support Vector Machine')

	support_vector_model = SGDClassifier(loss='hinge', penalty='l2',
		alpha=1e-3, n_iter=5)

	support_vector_model.fit(X_train_tfidf, y_train)

	predicted_svm = support_vector_model.predict(X_test_tfidf)
	
	print y_test
	print predicted_svm

	print('Accuracy: %f' % numpy.mean(predicted_svm == y_test))

	mostInformativeFeatures(support_vector_model, count_vect,[0,1,2,3],10)
	
	reliability = scipy.stats.pearsonr(predicted_svm,y_test)

	print (reliability)




if __name__ == '__main__':
	main()