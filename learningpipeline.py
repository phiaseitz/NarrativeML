import readnarratives
import scoresentencesdata
import visualizeresults
import numpy
import random
import scipy
import math
import mord
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier,Ridge,RidgeCV,LinearRegression
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import roc_auc_score

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
	"""This takes in a regression classfier and spits out the n features with 
	the higest and lowest features."""
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
	"""This takes in the scores for each sentence of the narrative and gives
	the whole narrative a score based on that"""
	non_zero = [score for score in scores if score != 0]

	average = sum(non_zero)/len(non_zero)

	if average < 0:
		return -1
	else:
		return 1


def splitTestTrain(X,y, train_size = 0.66, random_state = 42):
	"""Splitting X and y into test and train data"""
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
	"""Predict a score for each sentence based one the probabilites of the 
	scores"""
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
	"""Find the area underneath the ROC curve for different values
	 of the numberfeatures and the value for C (both in args_for_min)"""
	max_feat = abs(int(args_for_min[0]))
	C = abs(args_for_min[1])
	max_ngrams = abs(int(args_for_min[2]))
	#Read Files

	#count_vect = CountVectorizer(stop_words = "english")
	#count_vect = CountVectorizer(ngram_range=(1, 2),stop_words = "english")
	#count_vect = CountVectorizer(ngram_range=(1, 2))
	#count_vect = CountVectorizer()
	
	if max_ngrams == 1:
		tfidf_vect = TfidfVectorizer(token_pattern = r'\w*', 
		max_features = max_feat)
	else:
		tfidf_vect = TfidfVectorizer(token_pattern = r'\w*', 
		max_features = max_feat, ngram_range = (1,max_ngrams))

	X_train_tfidf = tfidf_vect.fit_transform(X_train)
	X_test_tfidf = tfidf_vect.fit_transform(X_test)

	#Logistic Regresssion
	#print('\n Logistic Regression \n')
	logistic_model = LogisticRegression(penalty = 'l2',
		tol = 0.00001, C=C, intercept_scaling=10000)
	logistic_model.fit(X_train_tfidf,y_train)
	
	#predicted_logistic = logistic_model.predict(X_test_tfidf)
	probs_logistic = logistic_model.predict_proba(X_test_tfidf)

	# Tell pos apart from neg
	roc_area = roc_auc_score([x == 1 for x in y_test if x != 0],
		[prob[2]-prob[0] for i,prob in enumerate(probs_logistic) 
		if y_test[i] != 0])

	#Tell one apart from others
	# roc_area = roc_auc_score([x == -1 for x in y_test],
	# 	[prob[0] for i,prob in enumerate(probs_logistic)])

	print "Maximum Features: {},C: {}, Ngrams: {} ".format(
		max_feat, C,max_ngrams)
	# print roc_area
	# print '\n'

	return 1-roc_area


		#print "Maximum Features: {} and C: {}".format(max_feat, C)
		
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


	#opvals = scipy.optimize.minimize(ROCArea, (450, 0.001,7), args = (
		# X_train, y_train, X_test, y_test), method = 'Nelder-Mead')
	roc_max = 0
	best_feats = 0
	best_C = 0
	best_ngrams = 0

	roc_min = 1
	worst_feats = 0
	worst_C = 0
	worst_ngrams = 0

	max_feats = range (300, 500, 10)
	max_feats_a = numpy.asarray(max_feats)
	c_vals = numpy.logspace(-4,-1,20).tolist()
	c_vals_a = numpy.asarray(c_vals)
	ngram_range = range(1,5)
	ngram_range_a = numpy.asarray(ngram_range)

	ROC = numpy.zeros((len(max_feats),len(c_vals), len(ngram_range)))

	for i,max_feat in enumerate(max_feats):
		for j,c_val in enumerate(c_vals):
			for k,ngram in enumerate(ngram_range):
				roc_temp = 1-ROCArea((max_feat,c_val,ngram), 
					X_train, y_train, X_test, y_test)
				ROC[i,j,k] = roc_temp
				# print ROC
				if roc_temp > roc_max:
					roc_max = roc_temp
					best_feats = max_feat
					best_C = c_val
					best_ngrams = ngram
				if roc_temp < roc_min:
					roc_min	 = roc_temp
					worst_feats = max_feat
					worst_C = c_val
					worst_ngrams = ngram
			print (
				"Best Feats: {}, Best C: {}, Best Ngrams: {} ".format(
					best_feats, best_C,best_ngrams))	
			print (roc_max)
	print ("Best Feats: {}, Best C: {}, Best Ngrams: {} ".format(
					best_feats, best_C,best_ngrams))	
	print (roc_max)

	levels = MaxNLocator(nbins=15).tick_values(roc_min, roc_max)
	cmap = plt.get_cmap('YlGnBu')
	norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)


	plt.subplot(2, 2, 1)
	plt.pcolor(numpy.log10(c_vals_a),max_feats_a,ROC[:,:,0],
		cmap = cmap, norm = norm)
	plt.title('1 grams')
	# set the limits of the plot to the limits of the data
	plt.axis([numpy.log10(c_vals_a).min(), numpy.log10(c_vals_a).max(), 
		max_feats_a.min(), max_feats_a.max()])
	plt.colorbar()

	plt.subplot(2, 2, 2)
	plt.pcolor(numpy.log10(c_vals_a),max_feats_a,ROC[:,:,1],
		cmap = cmap, norm = norm)
	plt.title('2 grams')
	# set the limits of the plot to the limits of the data
	plt.axis([numpy.log10(c_vals_a).min(), numpy.log10(c_vals_a).max(), 
		max_feats_a.min(), max_feats_a.max()])
	plt.colorbar()

	plt.subplot(2, 2, 3)
	plt.pcolor(numpy.log10(c_vals_a),max_feats_a,ROC[:,:,2],
		cmap = cmap, norm = norm)
	plt.title('3 grams')
	# set the limits of the plot to the limits of the data
	plt.axis([numpy.log10(c_vals_a).min(), numpy.log10(c_vals_a).max(), 
		max_feats_a.min(), max_feats_a.max()])
	plt.colorbar()

	plt.subplot(2, 2, 4)
	plt.pcolor(numpy.log10(c_vals_a),max_feats_a,ROC[:,:,3],
		cmap = cmap, norm = norm)
	plt.title('4 grams')
	# set the limits of the plot to the limits of the data
	plt.axis([numpy.log10(c_vals_a).min(), numpy.log10(c_vals_a).max(), 
		max_feats_a.min(), max_feats_a.max()])
	plt.colorbar()

	
	plt.show()

	# print(opvals.x)
	# print(1- ROCArea(opvals.x,X_train, y_train, X_test, y_test))


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