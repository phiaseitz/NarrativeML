import numpy
from scipy.sparse import csr_matrix
import hydrat
from hydrat.classifier.abstract import Learner, Classifier
from hydrat.task.sampler import stratify, allocate
from hydrat.task.taskset import from_partitions
from hydrat.experiments import Experiment
from hydrat.datamodel import FeatureMap, ClassMap, DataTaskSet

# TODO: Make the metaclassifier respect sequence info- right now it ignores it
# TODO: Make it possible to assemble the stackingclassifier from TaskSetResults directly
#       may want to write a function from_tsr, which takes a list of tsr and a metalearner.
#       Would still need some access to the l0 classifiers though.

class StackingL(Learner):
  __name__ = 'stacking'
  def __init__(self, metalearner, learner_committee, folds=10, rng=hydrat.rng):
    Learner.__init__(self)
    self.metalearner= metalearner 
    self.learner_committee= learner_committee 
    self.folds = folds
    self.rng = rng

  def _params(self):
    #TODO: RNG state?
    return dict( folds = self.folds
               , l0 = self.metalearner.__name__
               , l0_params = self.metalearner.params
               , l1 = dict((l.__name__, l.params) for l in self.learner_committee)
               )

  def _learn(self, feature_map, class_map):
    # generate a cross-validation over the training data.
    # TODO: refactor against dataset.split
    #
    strata_map = stratify(class_map)
    partition_proportions = numpy.array([1] * self.folds )
    parts  = allocate( strata_map
                     , partition_proportions
                     , probabilistic = False
                     , rng=self.rng
                     ) 
    parts = numpy.dstack((numpy.logical_not(parts), parts))
    fm = FeatureMap(feature_map, parts)
    cm = ClassMap(class_map, parts)
    taskset = DataTaskSet(fm, cm)

    # run each learner over the taskset, and produce a classification for each
    # instance-learner pair
    cl_feats = []
    for learner in self.learner_committee:
      experiment = Experiment(taskset, learner)
      tsr = experiment.run()
      cl = tsr.overall_classification(numpy.arange(feature_map.shape[0])).sum(axis=2)
      cl_feats.append(cl)
    cl_feats = csr_matrix(numpy.hstack(cl_feats))
    
    # train the metaclassifier
    metaclassifier = self.metalearner(cl_feats, class_map)
    classif_committee = [ l(feature_map, class_map) for l in self.learner_committee ]
    return StackingC(metaclassifier, classif_committee, self.__name__)
      

class StackingC(Classifier):
  def __init__(self, metaclassifier, classif_committee, name):
    self.__name__ = name
    Classifier.__init__(self)
    self.metaclassifier = metaclassifier
    self.classif_committee = classif_committee 

  def _classify(self, feature_map):
    pred_feats = []
    for c in self.classif_committee:
      pred_feats.append(c(feature_map))
    pred_feats = csr_matrix(numpy.hstack(pred_feats))
    return self.metaclassifier(pred_feats)
