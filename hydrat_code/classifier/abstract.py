"""
This module specifies the abstract interface that all classifier modules should
implement, and also provides some convenience methods.
"""
import numpy as n
import logging
import time
import inspect
import cPickle

import scipy.sparse as sp
import numpy as np

class ClassifierError(Exception): pass
class NoClassLabelsError(ClassifierError): pass
class NoFeaturesError(ClassifierError): pass
class NotInstalledError(ClassifierError): pass

def train(learner, fm, cm, kwargs):
  return learner(fm, cm, **kwargs)

def predict(classifier, fm, kwargs):
  return classifier(fm, **kwargs)

class Learner(object):
  def __init__(self):
    self.logger = logging.getLogger(__name__+'.'+self.__name__)
    self._check_installed()

  @property
  def metadata(self):
    md = {}
    md['learner']          = self.__name__
    md['learner_params']   = self.params
    return md

  def async(self, pool, feature_map, class_map, **kwargs):
    """
    Use a multiprocessing worker pool to train a classifier asynchronously.
    """
    return pool.apply_async(train, (self, feature_map, class_map, kwargs))

  def is_pickleable(self):
    """
    Checks that a learner and the corresponding classifier produced are both
    pickleable. We do this by testing it. This is needed to allow us to transmit
    learners and classifiers for multiprocessing. Learners can also override this 
    to avoid the need to test.
    
    TODO: There are situations where a pickleable learner that produces an
    unpickleable classifier are just fine, e.g. in experiment parallelization,
    where only the learner and result need to be pickleable. How do we
    handle this?
    """
    try:
      cPickle.dumps(self)
      fv = sp.csr_matrix([[1,0],[0,1]])
      cv = np.array([[1,0],[0,1]], dtype=bool)
      # Generate a trivial classifier
      cPickle.dumps(self(fv,cv))
      return True
    except (cPickle.UnpickleableError, TypeError):
      return False
    except Exception, e:
      self.logger.warning("Unexpected: {0} {1}".format(type(e), e))
      return False

  def __call__(self, feature_map, class_map, **kwargs):
    """
    We accept additional kwargs as a means to pass-through information.
    Meta classifiers should pass through kwargs.
    """
    num_docs, num_classes = class_map.shape
    num_features = feature_map.shape[1]
    self.logger.debug\
      ( "learning on %d documents in %d classes with %d features"
      , num_docs
      , num_classes
      , num_features
      ) 
    # We use introspection to determine which kwargs to pass through to the learner.
    # This is done to simplify the job of the programmer implementing the Learner 
    # subclass. The learner thus declares what additional information it is interested
    # in its _learn definition. Note that the value passed for any given keyword
    # could still be None.
    argspec = inspect.getargspec(self._learn)
    self.logger.debug( "argspec: %s", argspec)

    if argspec.keywords is not None:
      # if kwargs is bound in the learner, this means that the learner needs to be able
      # to pass through kwargs, so we must supply it with all the kwargs we have.
      supported_kwargs = dict(kwargs)
      self.logger.debug( "learner binds kwargs, all args passed" )
    else:
      supported_kwargs = dict()
      for key in argspec.args[3:]:
        supported_kwargs[key] = kwargs[key]
      self.logger.debug( "suported keywords: %s", supported_kwargs.keys() )

    start = time.time()
    classifier = self._learn(feature_map, class_map, **supported_kwargs)
    timetaken = time.time() - start
    classifier.metadata.update(self.metadata)
    classifier.metadata['learn_time']       = timetaken
    classifier.metadata['train_feat_count'] = num_features
      
    self.logger.debug("learning took %.1f seconds", timetaken)
    return classifier

  def _check_installed(self):
    """ Check that any external tools required are actually installed 
    Should raise an exception if they are not, and not return anything if they are
    """
    self.logger.warning("Learner '%s' does not implement _check_installed", self.__name__)

  @property
  def params(self):
    try:
      return self._params()
    except NotImplementedError:
      self.logger.warning("Learner '%s' does not implement _params", self.__name__)
      return None

  @property
  def desc(self):
    raise DeprecationWarning
    return self.__name__, self._params()

  def _learn(self, feature_map, class_map):
    """ Train a classifier
        Returns a Classifier object
    """
    raise NotImplementedError

  def _params(self):
    """
    Returns a dictionary describing the learner
    Ideally should be able to pass this to the learner class' __init__ **kwargs
    """
    raise NotImplementedError


class Classifier(object):
  def __init__(self):
    self.logger   = logging.getLogger("hydrat.classifier.%s"%self.__name__)
    self.metadata = { 'classifier' : self.__name__ }

  def __call__(self, feature_map, **kwargs):
    self.logger.debug("classifying %d documents", feature_map.shape[0])
    cl_num_feats = feature_map.shape[1]
    tr_num_feats = self.metadata['train_feat_count']
    if cl_num_feats != tr_num_feats: 
      raise ValueError, "Trained on %d features, %d for classification" % ( tr_num_feats
                                                                          , cl_num_feats
                                                                          )

    # Check that we have not provided any empty instances
    for i,row in enumerate(feature_map):
      if len(row.nonzero()) == 0:
        self.logger.warning("Empty classification instance at index %d!", i)

    argspec = inspect.getargspec(self._classify)
    self.logger.debug( "argspec: %s", argspec)

    if argspec.keywords is not None:
      # if kwargs is bound in the learner, this means that the learner needs to be able
      # to pass through kwargs, so we must supply it with all the kwargs we have.
      supported_kwargs = dict(kwargs)
      self.logger.debug( "classifier binds kwargs, all args passed" )
    else:
      supported_kwargs = dict()
      for key in argspec.args[2:]:
        supported_kwargs[key] = kwargs[key]
      self.logger.debug( "suported keywords: %s", supported_kwargs.keys() )

    start                           = time.time()
    classifications                 = self._classify(feature_map, **supported_kwargs)
    timetaken                       = time.time() - start
    self.metadata['classify_time']  = timetaken
    self.logger.debug("classification took %.1f seconds", timetaken)
    return classifications 

  def is_pickleable(self):
    try:
      cPickle.dumps(self)
      return True
    except (cPickle.UnpickleableError, TypeError):
      return False
    except Exception, e:
      self.logger.warning("Unexpected: {0} {1}".format(type(e), e))
      return False

  def async(self, pool, feature_map, **kwargs):
    return pool.apply_async(predict, (self, feature_map, kwargs))
    
  def _classify(self, feature_map):
    """ Classify a set of documents represented as an array of features
        Returns a boolean array:
          axis 0: document index
          axis 1: classlabels
    """
    raise NotImplementedError

class NullLearner(Learner):
  __name__ = "nulllearner"

  def __init__(self, classifier_constructor, name, *args, **kwargs):
    self.__name__ = name 
    Learner.__init__(self)
    assert issubclass(classifier_constructor, LearnerlessClassifier)
    self._args                   = args
    self._kwargs                 = kwargs
    self.classifier_constructor  = classifier_constructor

  def _learn(self, feature_map, class_map ):
    return self.classifier_constructor( self.__name__
                                      , feature_map
                                      , class_map
                                      , *self._args
                                      , **self._kwargs
                                      )

class LearnerlessClassifier(Classifier):
  def __init__(self, name, feature_map, class_map):
    self.__name__ = name
    Classifier.__init__(self)
    self.train_fv  = feature_map
    self.train_cv  = class_map
