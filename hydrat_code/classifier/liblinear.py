"""
Interface to liblinear
Based on libsvm.py from hydrat

Marco Lui
February 2012
"""
import tempfile
import os
import logging
import time
import numpy
import sys
import subprocess

from itertools import izip
from contextlib import closing

from hydrat import config
from hydrat.configuration import Configurable, EXE
from hydrat.classifier.abstract import Learner, Classifier, NotInstalledError
from hydrat.classifier.libsvm import SVMFileWriter

logger = logging.getLogger(__name__)
tempfile.tempdir = config.getpath('paths','scratch')

class liblinearL(Configurable, Learner):
  """
 -s type : set type of solver (default 1)
  0 -- L2-regularized logistic regression (primal)
  1 -- L2-regularized L2-loss support vector classification (dual)
  2 -- L2-regularized L2-loss support vector classification (primal)
  3 -- L2-regularized L1-loss support vector classification (dual)
  4 -- multi-class support vector classification by Crammer and Singer
  5 -- L1-regularized L2-loss support vector classification
  6 -- L1-regularized logistic regression
  7 -- L2-regularized logistic regression (dual)
-c cost : set the parameter C (default 1)
-e epsilon : set tolerance of termination criterion
  -s 0 and 2
    |f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,
    where f is the primal function and pos/neg are # of
    positive/negative data (default 0.01)
  -s 1, 3, 4 and 7
    Dual maximal violation <= eps; similar to libsvm (default 0.1)
  -s 5 and 6
    |f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,
    where f is the primal function (default 0.01)
-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)
-wi weight: weights adjust the parameter C of different classes (see README for details)
-v n: n-fold cross validation mode
-q : quiet mode (no outputs)
 """
  __name__ = 'liblinear'
  requires =\
    { ('tools', 'liblinearlearner')    : EXE('train')
    , ('tools','liblinearclassifier')  : EXE('predict')
    }

  def __init__(self, output_probability=False, 
      svm_type=1, cost=1, additional=''):
    self.clear_temp = config.getboolean('debug', 'clear_temp_files')

    if svm_type not in range(8):
      raise ValueError, 'unknown svm_type'

    self.output_probability = output_probability
    self.svm_type = svm_type
    self.cost = cost
    self.additional = additional

    self.learner = config.getpath('tools', 'liblinearlearner')
    self.classifier = config.getpath('tools','liblinearclassifier') 
    Learner.__init__(self)

    self.model_path = None
    self.clear_temp = config.getboolean('debug', 'clear_temp_files')

    self.__params = dict( output_probability=output_probability
                        , svm_type=svm_type
                        , cost=cost
                        , additional=additional
                        )

  def is_picklable(self):
    return True

  def __getstate__(self):
    return (self.output_probability, self.svm_type, self.cost, self.additional)

  def __setstate__(self, value):
    self.__init__(*value)

  def __del__(self):
    if self.clear_temp:
      if self.model_path is not None: os.remove(self.model_path)

  def _check_installed(self):
    if not all([os.path.exists(tool) for tool in 
        (self.learner, self.classifier, )]):
      self.logger.error("Tool not found for %s", self.__name__)
      raise NotInstalledError, "Unable to find required binary"

  def _params(self):
    return self.__params

  def _learn(self, feature_map, class_map):
     #Create and write the training file
    train = tempfile.NamedTemporaryFile(delete=self.clear_temp)
    self.logger.debug("writing training file: %s", train.name)
    # TODO: There is a potential speedup from reordering features,
    # such that we omit all non-zero features. The outcome should
    # be the same, the only difference is in obtaining theta.
    SVMFileWriter.write(train, feature_map, class_map)
    train.flush()

    #Create a temporary file for the model
    model_file, model_path = tempfile.mkstemp()
    self.logger.debug("model path: %s", model_path)
    os.close(model_file)

    # train svm
    if self.additional:
      training_cmd = [ self.learner, '-q', '-s', str(self.svm_type), '-c', str(self.cost)] +\
        self.additional.split() +\
        [ train.name , model_path ]
    else:
      training_cmd = [ self.learner, '-q', '-s', str(self.svm_type), '-c', 
          str(self.cost), train.name , model_path ]
    self.logger.debug("Training liblinear: %s", ' '.join(training_cmd))
    retcode = subprocess.call(training_cmd)
    if retcode:
      self.logger.critical("Training liblinear failed")
      raise ValueError, "Training liblinear returned %s"%(str(retcode))

    with open(model_path) as f:
      model_len = len(f.read())
      if model_len == 0:
        self.logger.critical("Training liblinear produced empty file")
        raise ValueError("Training liblinear produced empty file")
        
    return SVMClassifier( model_path, self.classifier, class_map.shape[1],
        probability_estimates=self.output_probability )

class SVMClassifier(Classifier):
  __name__ = "liblinear"

  def __init__(self, model_path, classifier, num_classes, 
      probability_estimates=False):
    Classifier.__init__(self)
    self.model_path = model_path
    self.classifier = classifier
    self.num_classes = num_classes
    self.probability_estimates = probability_estimates
    self.clear_temp = config.getboolean('debug','clear_temp_files')

  def is_picklable(self):
    return True

  def __getstate__(self):
    # we actually read the model file, as each instance
    # of the classifier is responsible for deleting its own
    # copy thereof. This means that for pickling, just having
    # a path is insufficient.
    with open(self.model_path) as f:
      model = f.read()

    initargs = (self.classifier, self.num_classes, self.probability_estimates)
    metadata = self.metadata
    return (initargs, metadata, model)

  def __setstate__(self, state):
    initargs, metadata, model = state

    _f, model_path = tempfile.mkstemp()
    os.close(_f)
    with open(model_path,'w') as f:
      f.write(state[2])

    initargs = (model_path,) + initargs
    self.__init__(*initargs)
    self.metadata.update(metadata)

  def __del__(self):
    if self.clear_temp:
      if self.model_path is not None: 
        os.remove(self.model_path)

  def __invoke_classifier(self, test_path):
    #Create a temporary file for the results
    result_file, result_path = tempfile.mkstemp()
    os.close(result_file)

    classif_cmd = [
        self.classifier, '-b', '1' if self.probability_estimates else '0', 
        test_path, self.model_path, result_path ]
    self.logger.debug("Classifying liblinear: %s", ' '.join(classif_cmd))
    retcode = subprocess.call(classif_cmd, stdout=open('/dev/null'))
    if retcode:
      self.logger.critical("Classifying liblinear failed")
      raise ValueError, "Classif liblinear returned %s"%(str(retcode))

    return result_path

  def __parse_result(self, result_path, num_test_docs):
    result_file = open(result_path)
    first_line = result_file.next()

    # Decide the type of file we are dealing with
    if first_line.split()[0] == 'labels':
      class_indices = map(int,first_line.split()[1:])
      self.logger.debug('Parsing libSVM probability output')
      classifications = numpy.zeros((num_test_docs, self.num_classes), dtype='float')
      for i, l in enumerate(result_file):
        # Read a list of floats, skipping the first entry which is the predicted class
        # We clip the lower values as liblinear can return really small probabilities.
        # This can be a problem downstream e.g. in stacking.
        # TODO: make the lower bound configurable?
        classifications[i, class_indices] = numpy.clip(map(float,l.split()[1:]), 1.0e-50, 1.0 )
          
    else:
      self.logger.debug('Parsing svm one-of-m output')
      classifications = numpy.zeros((num_test_docs, self.num_classes), dtype='bool')
      # Set the first class from what we earlier extracted.
      classifications[0, int(first_line)] = True
      for doc_index, l in enumerate(result_file):
        # TODO: Work out libSVM splits out class labels that are clearly out of the range
        #       of possible values. Seen it happen in multiclass contexts.
        class_index = int(l)
        classifications[doc_index+1, class_index] = True

    # Dispose of the unneeded output file
    result_file.close()
    if self.clear_temp:
      os.remove(result_path)

    return classifications

  @property
  def theta(self):
    """
    Read the parameter vector from the model file
    """
    retval = numpy.loadtxt(self.model_path, skiprows=6)
    if len(retval.shape) == 1:
      retval = retval[:,None]
    return retval

  def _classify(self, feature_map):
    test  = tempfile.NamedTemporaryFile(delete=self.clear_temp)
    self.logger.debug("writing test file: %s", test.name)
    SVMFileWriter.write(test, feature_map)
    test.flush()

    num_test_docs = feature_map.shape[0]
    result_path = self.__invoke_classifier(test.name)
    classifications = self.__parse_result(result_path, num_test_docs)
    return classifications
