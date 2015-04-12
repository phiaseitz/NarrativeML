"""
Example for a basic in-domain text classification task,
using both train/test and crossvalidation
"""

import hydrat.corpora.dummy as dummy
from hydrat.dataset.split import TrainTest, CrossValidation

import hydrat.classifier.NLTK as nltk
import hydrat.classifier.knn as knn
import hydrat.classifier.nearest_prototype as np
import hydrat.classifier.maxent as maxent
import hydrat.classifier.scikits_learn as scikits_learn
import hydrat.classifier.naivebayes as nb
#import hydrat.classifier.flann as flann

from hydrat.proxy import DataProxy
from hydrat.store import Store
from hydrat.experiment import Experiment

class unicode_dummy(dummy.unicode_dummy, TrainTest, CrossValidation): pass

learners = [
  np.cosine_mean_prototypeL(),
  knn.cosine_1nnL(),
  knn.skew_1nnL(),
  knn.oop_1nnL(),
  maxent.maxentLearner(),
  nltk.naivebayesL(),
  nltk.decisiontreeL(),
  scikits_learn.SVC(),
  scikits_learn.SVC(kernel='rbf'),
  scikits_learn.NuSVC(),
  scikits_learn.LinearSVC(),
  np.skew_mean_prototypeL(),
  nb.multinomialL(),
  #, flann.FLANNL()
  #, flann.kl()
  #, flann.cs()
  ]

datasets = [
  unicode_dummy(10),
  unicode_dummy(20),
  unicode_dummy(30),
  unicode_dummy(40),
  ]

features=[
  "byte_bigram",
  "byte_trigram",
  "byte_unigram",
  ]


if __name__ == "__main__":
  store = Store.from_caller()

  for feature in features:
    for ds in datasets:
      proxy = DataProxy(ds, store=store)
      proxy.class_space = 'dummy_default'
      proxy.feature_spaces = feature

      for l in learners:
        e = Experiment(proxy, l)
        r = store.new_TaskSetResult(e)
