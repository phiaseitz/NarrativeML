import readnarratives
import scoresentencesdata
import visualizeresults
import numpy
import random
import scipy
import math
import mord
import hydrat_code.classifier.ordinal as classifier
from sklearn import metrics

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier,Ridge,RidgeCV,LinearRegression
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import roc_auc_score
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

def scoreNarrativeFromSentences (scores):
	non_zero = [score for score in scores if score != 0]

	average = sum(non_zero)/len(non_zero)

	if average < 0:
		return -1
	else:
		return 1


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

def predictFromProbs(probs):
	problow = [prob[0] for prob in probs]
	probno = [prob[1] for prob in probs]
	probhigh = [prob[2] for prob in probs]

	avglow = sum(problow)/len(problow)
	avgno = sum(probno)/len(probno)
	avghigh = sum(probhigh)/len(probhigh)

	# print (avglow)
	# print (avgno)
	# print (avghigh)

	scores = []

	for prob_list in probs:
		lowdiff = (prob_list[0]-avglow)/avglow
		nodiff = (prob_list[1]-avgno)/avgno
		highdiff = (prob_list[2]-avghigh)/avghigh

		# print lowdiff
		# print nodiff
		# print highdiff

		if lowdiff > nodiff and lowdiff > highdiff:
			scores.append(-1)
		elif highdiff > nodiff and highdiff > lowdiff:
			scores.append(1)
		else:
			scores.append(0)

	return scores

	#print(avglow+avgno+avghigh)
def ROCArea(args_for_min, X_train, y_train, X_test, y_test):
	max_feat = abs(int(args_for_min[0]))
	C = abs(args_for_min[1])
	#Read Files

	#count_vect = CountVectorizer(stop_words = "english")
	#count_vect = CountVectorizer(ngram_range=(1, 2),stop_words = "english")
	#count_vect = CountVectorizer(ngram_range=(1, 2))
	#count_vect = CountVectorizer()
	

	tfidf_vect = TfidfVectorizer(token_pattern = r'\w*', 
		max_features = max_feat)

	X_train_tfidf = tfidf_vect.fit_transform(X_train)
	X_test_tfidf = tfidf_vect.fit_transform(X_test)

		#Logistic Regresssion
	#print('\n Logistic Regression \n')
	logistic_model = LogisticRegression(penalty = 'l2',
		tol = 0.00001, C=C, intercept_scaling=10000)
	logistic_model.fit(X_train_tfidf,y_train)
	
	#predicted_logistic = logistic_model.predict(X_test_tfidf)
	probs_logistic = logistic_model.predict_proba(X_test_tfidf)

	roc_area = roc_auc_score([x == 1 for x in y_test if x != 0],
		[prob[2]-prob[0] for i,prob in enumerate(probs_logistic) 
		if y_test[i] != 0])

	return 1-roc_area

	# 	print "Maximum Features: {} and C: {}".format(max_feat, C)
		
	# 	print roc_area

	# 	print '\n'

	# print 'Best Feature Num: {} and Best C: {}'.format(best_feats, best_C)
	# print 'Highest ROC area: {}'.format(max_area)



def main():

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

	numpy.set_printoptions(threshold=numpy.nan)


	opvals = scipy.optimize.minimize(ROCArea, (450, 0.001), args = (
		X_train, y_train, X_test, y_test), method = 'Nelder-Mead')

	print(opvals.x)
	print(1- ROCArea(opvals.x,X_train, y_train, X_test, y_test))


	# #Read Files
	# pickle_name = 'NarrativePickleAgency_test'
	# responses = readnarratives.readPickle(pickle_name)

	# sentences = scoresentencesdata.getSentenceData(responses)

	# texts = [sentence[0] for sentence in sentences]
	# scores_actual = [sentence[1] for sentence in sentences]

	# scores_dict = {}
	# scores_dict[0.0] = -1
	# scores_dict[1.0] = -1
	# scores_dict[1.5] = 0
	# scores_dict[2.0] = 1
	# scores_dict[3.0] = 1

	# scores_no_dec = [scores_dict[score] for score in scores_actual]

	# #Split test train
	# X_train, X_test, y_train, y_test = splitTestTrain(
	# 	texts, scores_no_dec, 0.66, 42)

	# #count_vect = CountVectorizer(stop_words = "english")
	# #count_vect = CountVectorizer(ngram_range=(1, 2),stop_words = "english")
	# #count_vect = CountVectorizer(ngram_range=(1, 2))
	# #count_vect = CountVectorizer()
	
	# max_feats = range(425,475,1)
	# Cs = [10**-2,10**-3,10**-4]

	# best_C = -1
	# best_feats = -1
	# max_area = -1

	# for max_feat in max_feats:
	# 	for C in Cs:
	# 		tfidf_vect = TfidfVectorizer(token_pattern = r'\w*', 
	# 			max_features = max_feat)

	# 		X_train_tfidf = tfidf_vect.fit_transform(X_train)
	# 		X_test_tfidf = tfidf_vect.fit_transform(X_test)

	# 			#Logistic Regresssion
	# 		#print('\n Logistic Regression \n')
	# 		logistic_model = LogisticRegression(penalty = 'l2',
	# 			tol = 0.00001, C=C, intercept_scaling=10000)
	# 		logistic_model.fit(X_train_tfidf,y_train)
			
	# 		#predicted_logistic = logistic_model.predict(X_test_tfidf)
	# 		probs_logistic = logistic_model.predict_proba(X_test_tfidf)

	# 		roc_area = roc_auc_score([x == 1 for x in y_test if x != 0],
	# 			[prob[2]-prob[0] for i,prob in enumerate(probs_logistic) 
	# 			if y_test[i] != 0])

	# 		if roc_area > max_area:
	# 			max_area = roc_area
	# 			best_C = C
	# 			best_feats = max_feat

	# 		print "Maximum Features: {} and C: {}".format(max_feat, C)
			
	# 		print roc_area

	# 		print '\n'

	# print 'Best Feature Num: {} and Best C: {}'.format(best_feats, best_C)
	# print 'Highest ROC area: {}'.format(max_area)

			#print y_test
			#print predicted_logistic

			# print probs_logistic

			# print (y_test - predicted_logistic)

			# print(logistic_model.score(X_test_tfidf,y_test))

			# mostInformativeFeatures(logistic_model, tfidf_vect,[-1,0,1],15)

			#print y_test
			#scores = predictFromProbs(probs_logistic)

	# print scores
	# print y_test
	# diffs = [scores[i]-y_test[i] for i in range(len(y_test))]
	# print ('Correctly predicted -1s')
	# print(sum([1 for i in range(len(scores)) 
	# 	if (y_test[i] == -1 and scores[i] == -1)])/float(y_test.count(-1)))
	# print ('Correctly predicted 0s')
	# print(sum([1 for i in range(len(scores)) 
	# 	if (y_test[i] == 0 and scores[i] == 0)])/float(y_test.count(0)))
	# print ('Correctly predicted 1s')
	# print(sum([1 for i in range(len(scores)) 
	# 	if (y_test[i] == 1 and scores[i] == 1)])/float(y_test.count(1)))
	# print ('Total')
	# print((sum([1 for x in diffs if x == 0])/float(len(diffs))))


	# print y_test
	#print [x == 1 for x in y_test]
	


if __name__ == '__main__':
	main()