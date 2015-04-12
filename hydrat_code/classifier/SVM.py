"""
External SVM classifiers using a command-line interface
TODO: Why does SVM using 'output probability' perform different from without?
TODO: Use pexpect instead of popen, tie in with progressbar output
"""
from hydrat.classifier.abstract import Learner, Classifier, NotInstalledError
from hydrat import config
from hydrat.configuration import Configurable, EXE

from itertools import izip

import tempfile
import os
import logging
import time
import numpy
import sys

tempfile.tempdir = config.getpath('paths','scratch')

__all__ =\
  [ "libsvmExtL"
  , "bsvmL"
  #, "f_libsvmL"
  #, "f_libsvmScaleL"
  #, "f_libsvmProbL" 
  ]

class SVMFileWriter(object):
  logger = logging.getLogger("hydrat.classifier.svmfilewriter")

  @staticmethod
  def instance(fv, cv = None):
    fv.sort_indices()
    indices = fv.indices  
    values = fv.data
    
    if cv is not None:
      classids = numpy.arange(cv.shape[0])[cv]
      if len(classids) == 0:
        classids = [-1] 
    else:
      classids = [-1] #If classes not known just write negative 
    classlabelblock = ",".join(str(id) for id in classids)
    features = " ".join(    str(indices[i]+1) + ':' + str(values[i]) 
                       for  i
                       in   xrange(len(indices)) 
                       )
    return "%s %s\n" % (classlabelblock,features)

  @staticmethod
  def writefile(file, fvs, cvs = None):
    if cvs is not None: assert fvs.shape[0] == cvs.shape[0] 
    for i in xrange(fvs.shape[0]):
      if cvs is not None:
        instance = SVMFileWriter.instance(fvs[i], cvs[i])
      else:
        instance = SVMFileWriter.instance(fvs[i])
      file.write(instance)


class SVMLearner(Learner):
  __name__ = "svm"
  KERNEL_TYPE_CODES = dict(linear=0, polynomial=1, rbf=2, sigmoid=3)

  def __init__(self):
    self.range_path = None
    self.model_path = None
    self.clear_temp = config.getboolean('debug', 'clear_temp_files')
    Learner.__init__(self)

  def _check_installed(self):
    def tool_ok(path):
      return os.path.exists(path)
    if not all([tool_ok(self.learner), tool_ok(self.classifier)]):
      self.logger.error("Tool not found for %s", self.__name__)
      raise NotInstalledError, "Unable to find required binary"
   
  def _params(self):
    return dict()

  def _learn(self, feature_map, class_map):
    writer = SVMFileWriter

     #Create and write the training file
    train = tempfile.NamedTemporaryFile(delete=self.clear_temp)
    self.logger.debug("writing training file: %s", train.name)
    writer.writefile(train, feature_map, class_map)
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
      scaling_command ='%s -l 0 -s "%s" "%s" > "%s"' % ( self.scaler
                                                       , self.range_path
                                                       , train.name
                                                       , scale.name
                                                       )
      self.logger.debug("Scaling SVM: %s", scaling_command)
      process = os.popen(scaling_command)
      output = process.read()
      return_value = process.close()
      if return_value:
        self.logger.critical("Scaling SVM failed with output:")
        self.logger.critical(output)
        raise ValueError, "Scaling SVM returned %s"%(str(return_value))


    train_path = scale.name if self.scale else train.name
    training_command =\
      "%s %s %s %s" % ( self.learner
                      , '-b 1' if self.output_probability else '' 
                      , train_path
                      , self.model_path 
                      )
    self.logger.debug("Training SVM: %s", training_command)
    process = os.popen(training_command)
    output = process.read()
    return_value = process.close()
    if return_value:
      self.logger.critical("Training SVM failed with output:")
      self.logger.critical(output)
      raise ValueError, "Training SVM returned %s"%(str(return_value))

    classif_opts = ' -b 1' if self.output_probability else ''
    return SVMClassifier( self.model_path
                        , self.classifier + classif_opts
                        , class_map.shape[1]
                        , self.scaler if self.scale else None
                        , self.range_path
                        )

  def __del__(self):
    if self.clear_temp:
      if self.model_path is not None: os.remove(self.model_path)
      if self.range_path is not None: os.remove(self.range_path)

class SVMClassifier(Classifier):
  __name__ = "svm"

  def __init__(self, model_path, classifier, num_classes, scaler = None, range_path = None):
    Classifier.__init__(self)
    self.model_path  = model_path
    self.classifier  = classifier
    self.scaler      = scaler
    self.range_path  = range_path
    self.num_classes = num_classes
    self.clear_temp  = config.getboolean('debug','clear_temp_files')

  def __invoke_classifier(self, test_path):
    #Create a temporary file for the results
    result_file, result_path = tempfile.mkstemp()
    os.close(result_file)

    if self.scaler is not None:
      scale = tempfile.NamedTemporaryFile(delete=self.clear_temp)
      self.logger.debug("scale path: %s", scale.name)

      self.logger.debug("scaling test data")
      scaling_command ='%s -r "%s" "%s" > "%s"' % ( self.scaler
                                                  , self.range_path
                                                  , test_path 
                                                  , scale.name
                                                  )
      self.logger.debug("Scaling SVM: %s", scaling_command)
      process = os.popen(scaling_command)
      output = process.read()
      return_value = process.close()
      if return_value:
        self.logger.critical("Scaling SVM failed with output:")
        self.logger.critical(output)
        raise ValueError, "Scaling SVM returned %s"%(str(return_value))

    test_path = test_path if self.scaler is None else scale.name
    classif_command = "%s %s %s %s" % ( self.classifier
                                      , test_path
                                      , self.model_path
                                      , result_path
                                      )
    self.logger.debug("Classifying SVM: %s", classif_command)
    process = os.popen(classif_command)
    output = process.read()
    return_value = process.close()
    if return_value:
      self.logger.critical("Classifying SVM failed with output:\n"+output)
      raise ValueError, "Classif SVM returned %s"%(str(return_value))

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

  def write_testfile(self, test, feature_map):
    writer = SVMFileWriter

     #Create and write the test file
    self.logger.debug("writing test file: %s", test.name)
    writer.writefile(test, feature_map)
    test.flush()

  def _classify(self, feature_map):
    test  = tempfile.NamedTemporaryFile(delete=self.clear_temp)
    self.write_testfile(test, feature_map)
    num_test_docs = feature_map.shape[0]

    return self.classify_from_file(test.name, num_test_docs)

  def classify_from_file(self, test_path, num_test_docs):
    result_path = self.__invoke_classifier(test_path)
    classifications = self.__parse_result(result_path, num_test_docs)
    return classifications

class libsvmExtL(Configurable, SVMLearner):
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
  __name__ = 'libsvm_ext'
  requires =\
    { ('tools', 'libsvmlearner')    : EXE('svm-train')
    , ('tools','libsvmclassifier')  : EXE('svm-predict')
    , ('tools','libsvmscaler')      : EXE('svm-scale')
    }

  def _check_installed(self):
    SVMLearner._check_installed(self)
    if not os.path.exists(self.scaler):
      self.logger.error("Unable to find required binaries")
      raise NotInstalledError, "Unable to find required binaries"


  def __init__(self, scale=False, output_probability=False, svm_type=0, kernel_type='rbf', additional=''):
    self.learner = config.getpath('tools', 'libsvmlearner')
    self.classifier = config.getpath('tools','libsvmclassifier') 
    self.scaler = config.getpath('tools', 'libsvmscaler')
    SVMLearner.__init__(self)
    self.scale = scale
    self.output_probability = output_probability
    self.__params = dict( scale=scale
                        , output_probability=output_probability
                        , kernel_type=kernel_type
                        , svm_type=svm_type
                        , additional=additional
                        )
    k = self.KERNEL_TYPE_CODES[kernel_type]

    # TODO assertions are not the right thing to use here
    assert svm_type in range(4), "svm_type value not acceptable!"
    assert k in range(4), "kernel_type value not acceptable!"
    self.learner += ' -s %d -t %d' % (svm_type, k)

  def _params(self):
    p = SVMLearner._params(self)
    p.update(self.__params)
    return p

# TODO: Deal with mutliclass fed to BSVM. Seems to choke, probably some error output on stderr.
class bsvmL(Configurable, SVMLearner):
  """
  -s svm_type : set type of SVM (default 0)
          0 -- multi-class bound-constrained support vector classification (SVC)
          1 -- multi-class SVC from solving a bound-constrained problem
          2 -- multi-class SVC from Cram and Singer
          3 -- bound-constrained support vector regression
  -t kernel_type : set type of kernel function (default 2)
          0 -- linear: u'*v
          1 -- polynomial: (gamma*u'*v + coef0)^degree
          2 -- radial basis function: exp(-gamma*|u-v|^2)
          3 -- sigmoid: tanh(gamma*u'*v + coef0)
  -d degree : set degree in kernel function (default 3)
  -g gamma : set gamma in kernel function (default 1/k)
  -r coef0 : set coef0 in kernel function (default 0)
  -c cost : set the parameter C of support vector machine (default 1)
  -p epsilon : set the epsilon in loss function of support vector regression (default 0.1)
  -m cachesize : set cache memory size in MB (default 100) (useless for -t 0)
  -e epsilon : set tolerance of termination criterion (default 0.001)
  -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
  -b cost0 : set the beginning cost value of alpha seeding for -t 0 -s 0 and 3 (default 1)
  -i step : set the step value of alpha seeding for -t 0 -s 0 and 3 (default 2)
                  cost0, cost0*step, cost0*step^2, ..., cost.
  -q qpsize : set the sub-problem size for -s 0,1 and 3
  -wi weight: set the parameter C of class i to weight*C, for classification problem (default 1)
  -v n: n-fold cross validation mode
  """
  __name__ = 'bsvm'
  requires =\
    { ('tools','bsvmlearner')    : EXE('bsvm-train')
    , ('tools','bsvmclassifier') : EXE('bsvm-predict')
    }


  def __init__(self, kernel_type='rbf', svm_type=0, additional=''):
    self.learner = config.getpath('tools', 'bsvmlearner')
    self.classifier = config.getpath('tools','bsvmclassifier') 
    SVMLearner.__init__(self)
    self.scale = False #No scaler for bsvm 
    self.output_probability = False # No probability output for bsvm
    self.__params = dict(kernel_type=kernel_type, svm_type=svm_type, additional=additional)
    k = self.KERNEL_TYPE_CODES[kernel_type]

    assert svm_type in range(4), "svm_type value not acceptable!"
    assert k in range(4), "kernel_type value not acceptable!"
    self.learner += ' -s %d -t %d' % (svm_type, k)

  def _params(self):
    p = SVMLearner._params(self)
    p.update(self.__params)
    return p

