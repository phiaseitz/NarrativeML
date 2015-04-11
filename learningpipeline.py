import readnarratives
import scoresentencesdata
import visualizeresults
import numpy
import random
import scipy
import math
import mord
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier,Ridge,RidgeCV,LinearRegression
from sklearn.linear_model import RidgeCV, LogisticRegression

def mostInformativeFeatures(classifier, vectorizer, 
	categories,number_of_features):
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
	#If we're doing regression, we don't have weights based on category
	#Then we print the most and least agentic words
	feature_names = numpy.asarray(vectorizer.get_feature_names())
	sorted_feats = numpy.argsort(classifier.coef_)
	#Last n
	top_pos = sorted_feats[-number_of_features:]
	#First n
	top_neg = sorted_feats[0:number_of_features]
	print("Top Positive: %s" % ", ".join(feature_names[top_pos]))
	print("Top Negative: %s" % ", ".join(feature_names[top_neg])) 

def splitTestTrain(X,y, train_size = 0.66, random_state = 42):
	#Making my own so I can split text!
	random.seed(random_state)
	shuffle_X = X
	shuffle_y = y
	#Shuffle the lists
	random.shuffle(shuffle_X)
	random.shuffle(shuffle_y)

	#Split basen on what proportion of test and train
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

	#count_vect = CountVectorizer(stop_words = "english")
	#count_vect = CountVectorizer(ngram_range=(1, 2),stop_words = "english")
	#count_vect = CountVectorizer(ngram_range=(1, 2))
	#count_vect = CountVectorizer()
	count_vect = CountVectorizer(token_pattern = r'\w*')

	X_train_count = count_vect.fit_transform(X_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)
	X_train_tfidf_a = tfidf_transformer.fit_transform(X_train_count).toarray()

	X_test_count = count_vect.transform(X_test)
	X_test_tfidf = tfidf_transformer.transform(X_test_count)
	X_test_tfidf_a = tfidf_transformer.transform(X_test_count).toarray()
	
	multiclass_logistic = mord.MulticlassLogistic(alpha = 100, verbose = 1)

	multiclass_logistic.fit(X_train_tfidf_a,y_train)

	predicted_mord = multiclass_logistic.predict(X_test_tfidf_a)

	print predicted_mord

	print (multiclass_logistic.score(X_test_tfidf_a, y_test))

	# #Naive Bayes - I'm not sure but this might not work so well
	# 	#always predicts 1...
	# print ('\nNaive Bayes \n')
	# #naive_bayes_model = BernoulliNB().fit(X_train_tfidf, y_train)
	# naive_bayes_model = MultinomialNB().fit(X_train_tfidf, y_train)

	# predicted_nb = naive_bayes_model.predict(X_test_tfidf)
	# #print('Accuracy: %f' % numpy.mean(predicted_nb == y_test))

	# print y_test
	# print predicted_nb

	# mostInformativeFeatures(naive_bayes_model, count_vect,[-1, 0, 1],15)

	# #reliability_nb = scipy.stats.pearsonr(predicted_nb,y_test)
	# print(naive_bayes_model.score(X_test_tfidf,y_test))

	# #print ('Accuracy: %f' % reliability_nb[0])
	# #Support vector
	# print('\nSupport Vector Machine \n')

	# support_vector_model = SGDClassifier(loss='hinge', penalty='l2',
	# 	alpha=1e-3, n_iter=4)

	# support_vector_model.fit(X_train_tfidf, y_train)

	# predicted_svm = support_vector_model.predict(X_test_tfidf)
	
	# print y_test
	# print predicted_svm

	# #print('Accuracy: %f' % numpy.mean(predicted_svm == y_test))

	# mostInformativeFeatures(support_vector_model, count_vect,[-1, 0,1],15)
	
	# reliability_svm = scipy.stats.pearsonr(predicted_svm,y_test)

	# #print ('Accuracy: %f' % reliability_svm[0])
	# print(support_vector_model.score(X_test_tfidf,y_test))

	# #Ridge Regression
	# print('\nRidge Regression \n')
	# ridge_model = Ridge(alpha = 10) #When only doing bag of words, 10 is good
	# ridge_model.fit(X_train_tfidf,y_train)

	# predicted_ridge = ridge_model.predict(X_test_tfidf)

	# #print(ridge_model.alpha_)
	# print y_test
	# print predicted_ridge
	# reliability_ridge = scipy.stats.pearsonr(predicted_ridge,y_test)

	# #print(ridge_model.coef_)

	# mostInformativeFeaturesRegression(ridge_model, count_vect,15)
	# print(ridge_model.score(X_test_tfidf,y_test))

	# raw_input("Press Enter to continue...")

	# to_print = range(len(X_test))

	# visualizeresults.visualizeWeightsList(to_print, X_test,X_test_tfidf,
	# 	y_test, count_vect,ridge_model)

	#Logistic ordinal regression
	# w, theta = logisticordinalregression.ordinal_logistic_fit(
	# 	X_train_tfidf_a, y_train, verbose=True, solver='TNC')
	# print theta	
	# pred = logisticordinalregression.ordinal_logistic_predict(w, theta, 
	# 	X_test_tfidf_a)
 #        s = metrics.mean_absolute_error(y_test, pred)

 #        print (pred)
 #        print (y_test)
 #        print('ERROR (ORDINAL)  fold %s: %s' % (i+1, s))



	# #Linear Regresssion

	# print('\nLinear Regression \n')
	# linear_model = LinearRegression()
	# linear_model.fit(X_train_tfidf,y_train)

	# predicted_linear = linear_model.predict(X_test_tfidf)

	# #print(ridge_model.alpha_)
	# print y_test
	# print predicted_linear
	# reliability_linear = scipy.stats.pearsonr(predicted_linear,y_test)

	# #print(ridge_model.coef_)

	# mostInformativeFeaturesRegression(linear_model, count_vect,15)
	# print(linear_model.score(X_test_tfidf,y_test))

	#print('Accuracy: %f' % reliability_linear[0])

	# 	#Linear Regresssion

	# print('\n Logistic Regression \n')
	# logistic_model = LogisticRegression(penalty = 'l1')
	# logistic_model.fit(X_train_tfidf,y_train)

	# predicted_logistic = linear_model.predict(X_test_tfidf)

	# print y_test
	# print predicted_linear

	# print(logistic_model.score(X_test_tfidf,y_test))

	# mostInformativeFeatures(logistic_model, count_vect,[0,1,2,3],15)

	# #print('Accuracy: %f' % reliability_linear[0])


if __name__ == '__main__':
	main()