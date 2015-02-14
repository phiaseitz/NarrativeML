import readnarratives
import numpy
import re

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

def tokenize(texts):
	count_vect = CountVectorizer()
	tokenized_texts = count_vect.fit_transform(texts)

	#print tokenized_texts.shape

	tfidf_transformer = TfidfTransformer()
	texts_tfidf = tfidf_transformer.fit_transform(tokenized_texts)

	return texts_tfidf,count_vect

def mostInformativeFeatures(classifier, vectorizer, categories):
	feature_names = numpy.asarray(vectorizer.get_feature_names())
	for i, category in enumerate(categories):
		top10 = numpy.argsort(classifier.coef_[i])[-10:]
		print("%s: %s" % (category, " ".join(feature_names[top10])))

def main():
	pickle_name = 'NarrativePickleAgency'
	texts, scores = readnarratives.readPickle(pickle_name)

	texts_tfidf,vectorizer = tokenize(texts)
	
	X_train, X_test, y_train, y_test = train_test_split(
		texts_tfidf, scores, test_size=0.33, random_state=42)

	naive_bayes_model = MultinomialNB().fit(X_train, y_train)

	predicted = naive_bayes_model.predict(X_test)
	print('Accuracy: %f' % numpy.mean(predicted == y_test))

	mostInformativeFeatures(naive_bayes_model, vectorizer,[0,1,2,3])


if __name__ == '__main__':
	main()