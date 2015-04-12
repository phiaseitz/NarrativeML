"""
External SVM classifiers using a command-line interface
TODO: Why does SVM using 'output probability' perform different from without?
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

logger = logging.getLogger(__name__)
tempfile.tempdir = config.getpath('paths','scratch')

class SVMFileWriter(object):
  """
  Output to a libsvm-compatible file format.
  """
  def __init__(self, fileh):
    if isinstance(fileh, basestring):
      self.fileh = open(fileh, 'w')
      self.is_path = True
    else:
      self.fileh = fileh
      self.is_path = False

  def instance(self, fv, cv = None):
    """
    Write a single instance to the file.
    """
    fv.sort_indices()
    indices = fv.indices  
    values = fv.data
    
    if cv is not None:
      classids = numpy.flatnonzero(cv)
      if len(classids) == 0:
        classids = [-1] # negative for no class
    else:
      classids = [-1] # negative for no class
    classlabelblock = ",".join(str(id) for id in classids)
    assert classlabelblock, "empty classlabel block"
    features = " ".join(    str(indices[i]+1) + ':' + str(values[i]) 
                       for  i
                       in   xrange(len(indices)) 
                       )
    self.fileh.write("%s %s\n" % (classlabelblock,features))

  def close(self):
    if self.is_path:
      self.fileh.close()
    else:
      self.fileh.flush()

  @classmethod
  def write(cls, path, fvs, cvs = None):
    if cvs is not None and (fvs.shape[0] != cvs.shape[0]):
      raise ValueError, "dimension mismatch"
    with closing(cls(path)) as f:
      for i in xrange(fvs.shape[0]):
        if cvs is not None:
          f.instance(fvs[i], cvs[i])
        else:
          f.instance(fvs[i])


class libsvmL(Configurable, Learner):
  """
  -s svm_type : set type of SVM (default 0)
          0 -- C-SVC
          1 -- nu-SVC
          2 -- one-class SVM
          3 -- epsilon-SVR
          4 -- nu-SVR
  -t kernel_type : set type of kernel function (default 2)
          0 -- linear: u'*v
          1 -- polynomial: (gamma*u'*v + coef0)^degree
          2 -- radial basis function: exp(-gamma*|u-v|^2)
          3 -- sigmoid: tanh(gamma*u'*v + coef0)
          4 -- precomputed kernel (kernel values in training_set_file)
  -d degree : set degree in kernel function (default 3)
  -g gamma : set gamma in kernel function (default 1/k)
  -r coef0 : set coef0 in kernel function (default 0)
  -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
  -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
  -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
  -m cachesize : set cache memory size in MB (default 100)
  -e epsilon : set tolerance of termination criterion (default 0.001)
  -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
  -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
  -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
  -v n: n-fold cross validation mode
  -q : quiet mode (no outputs)
  """
  __name__ = 'libsvm'
  requires =\
    { ('tools', 'libsvmlearner')    : EXE('svm-train')
    , ('tools','libsvmclassifier')  : EXE('svm-predict')
    , ('tools','libsvmscaler')      : EXE('svm-scale')
    }
  KERNEL_TYPE_CODES = dict(linear=0, polynomial=1, rbf=2, sigmoid=3)

  def __init__(self, scale=False, output_probability=False, 
      svm_type=0, kernel_type='rbf', additional=''):

    if svm_type not in range(5):
      raise ValueError, 'unknown svm_type'
    if kernel_type not in self.KERNEL_TYPE_CODES:
      raise ValueError, 'unknown kernel_type'

    self.scale = scale
    self.output_probability = output_probability
    self.svm_type = svm_type
    self.kernel_type = self.KERNEL_TYPE_CODES[kernel_type]

    self.learner = config.getpath('tools', 'libsvmlearner')
    self.classifier = config.getpath('tools','libsvmclassifier') 
    self.scaler = config.getpath('tools', 'libsvmscaler')
    Learner.__init__(self)

    self.range_path = None
    self.model_path = None
    self.clear_temp = config.getboolean('debug', 'clear_temp_files')

    self.__params = dict( scale=scale
                        , output_probability=output_probability
                        , kernel_type=kernel_type
                        , svm_type=svm_type
                        , additional=additional
                        )

  def __del__(self):
    if self.clear_temp:
      if self.model_path is not None: os.remove(self.model_path)
      if self.range_path is not None: os.remove(self.range_path)

  def _check_installed(self):
    if not all([os.path.exists(tool) for tool in 
        (self.learner, self.classifier, self.scaler)]):
      self.logger.error("Tool not found for %s", self.__name__)
      raise NotInstalledError, "Unable to find required binary"

  def _params(self):
    return self.__params

  def _learn(self, feature_map, class_map):
     #Create and write the training file
    train = tempfile.NamedTemporaryFile(delete=self.clear_temp)
    self.logger.debug("writing training file: %s", train.name)
    SVMFileWriter.write(train, feature_map, class_map)
    train.flush()

    #Create a temporary file for the model
    model_file, self.model_path = tempfile.mkstemp()
    self.logger.debug("model path: %s", self.model_path)
    os.close(model_file)

    # Perform scaling if requested
    if self.scale:
      scale = tempfile.NamedTemporaryFile(delete=self.clear_temp)
      self.logger.debug("scale path: %s", scale.name)

      range_file, self.range_path = tempfile.mkstemp()
      self.logger.debug("range path: %s", self.range_path)
      os.close(range_file)
      
      self.logger.debug("scaling training data")
      scaling_cmd = [self.scaler, '-l 0', '-s %s' % self.range_path, train.name]
      self.logger.debug("Scaling SVM: %s", ' '.join(scaling_cmd))
      retcode = subprocess.call(scaling_cmd, stdout=scale)
      if retcode:
        self.logger.critical("Scaling SVM failed")
        raise ValueError, "Scaling SVM returned %s"%(str(retcode))


    # train svm
    train_path = scale.name if self.scale else train.name
    training_cmd = [ self.learner, '-q', '-s', str(self.svm_type), '-t', 
        str(self.kernel_type), '-b', '1' if self.output_probability else '0', 
        train_path , self.model_path ]
    self.logger.debug("Training SVM: %s", ' '.join(training_cmd))
    retcode = subprocess.call(training_cmd)
    if retcode:
      self.logger.critical("Training SVM failed")
      raise ValueError, "Training SVM returned %s"%(str(retcode))

    return SVMClassifier( self.model_path, self.classifier, class_map.shape[1],
        scaler=self.scaler if self.scale else None, range_path=self.range_path,
        probability_estimates=self.output_probability )

class SVMClassifier(Classifier):
  __name__ = "svm"

  def __init__(self, model_path, classifier, num_classes, scaler = None, 
      range_path = None, probability_estimates=False):
    Classifier.__init__(self)
    self.model_path = model_path
    self.classifier = classifier
    self.num_classes = num_classes
    self.scaler = scaler
    self.range_path = range_path
    self.probability_estimates = probability_estimates
    self.clear_temp = config.getboolean('debug','clear_temp_files')

  def __invoke_classifier(self, test_path):
    #Create a temporary file for the results
    result_file, result_path = tempfile.mkstemp()
    os.close(result_file)

    if self.scaler is not None:
      scale = tempfile.NamedTemporaryFile(delete=self.clear_temp)
      self.logger.debug("scale path: %s", scale.name)

      self.logger.debug("scaling test data")
      scaling_cmd = [self.scaler, '-l', '0', '-r', self.range_path, test_path]

      self.logger.debug("Scaling SVM: %s", ' '.join(scaling_cmd))
      retcode = subprocess.call(scaling_cmd, stdout=scale)
      if retcode:
        self.logger.critical("Scaling SVM failed")
        raise ValueError, "Scaling SVM returned %s"%(str(retcode))

    test_path = test_path if self.scaler is None else scale.name
    classif_cmd = [
        self.classifier, '-b', '1' if self.probability_estimates else '0', 
        test_path, self.model_path, result_path ]
    self.logger.debug("Classifying SVM: %s", ' '.join(classif_cmd))
    retcode = subprocess.call(classif_cmd, stdout=open('/dev/null'))
    if retcode:
      self.logger.critical("Classifying SVM failed")
      raise ValueError, "Classif SVM returned %s"%(str(retcode))

    return result_path

  def __parse_result(self, result_path, num_test_docs):
    result_file = open(result_path)
    first_line = result_file.next()

    # Decide the type of file we are dealing with
    if first_line.split()[0] == 'labels':
      self.logger.debug('Parsing libSVM probability output')
      classifications = numpy.zeros((num_test_docs, self.num_classes), dtype='float')
      for i, l in enumerate(result_file):
        # Read a list of floats, skipping the first entry which is the predicted class
        classifications[i] = map(float,l.split()[1:])
          
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

  def _classify(self, feature_map):
    test  = tempfile.NamedTemporaryFile(delete=self.clear_temp)
    self.logger.debug("writing test file: %s", test.name)
    SVMFileWriter.write(test, feature_map)
    test.flush()

    num_test_docs = feature_map.shape[0]
    result_path = self.__invoke_classifier(test.name)
    classifications = self.__parse_result(result_path, num_test_docs)
    return classifications
