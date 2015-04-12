from hydrat.classifier.abstract import Learner, Classifier
from hydrat import config

import tempfile
import os
import numpy
import subprocess

from hydrat.classifier.libsvm import SVMFileWriter
from hydrat.configuration import is_exe, Configurable, EXE

"""
Wrapper for Le Zhang's maxent toolkit
http://homepages.inf.ed.ac.uk/lzhang10/maxent_toolkit.html
"""

class maxentLearner(Configurable, Learner):
  __name__ = "maxent"
  requires =\
    { ('tools','maxent') : EXE('maxent')
    }

  def __init__(self, iterations=3, method='lbfgs' ):
    self.toolpath = config.get('tools','maxent')
    Learner.__init__(self)
    if method not in ['lbfgs','gis']:
      raise ValueError, "Invalid method '%s'"%method
    self.iterations = iterations
    self.method = method
    self.model_path = None
    self.clear_temp = config.getboolean('debug','clear_temp_files')

  def _check_installed(self):
    if not is_exe(self.toolpath):
      raise ValueError, "Tool not installed!"

  def _params(self):
    return dict( iterations = self.iterations
              , method = self.method
              )

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

    train_path = train.name 
    training_cmd = [ self.toolpath, train_path, '-b', '-m', self.model_path, 
        '-i', str(self.iterations) ]
    self.logger.debug("Training maxent: %s", ' '.join(training_cmd))
    retcode = subprocess.call(training_cmd)
    if retcode:
      self.logger.critical("Training maxent failed")
      raise ValueError, "Training maxent returned %s"%(str(retcode))

    return maxentClassifier( self.model_path
                          , class_map.shape[1]
                          , self.__name__
                          )

  def __del__(self):
    if self.clear_temp:
      if self.model_path is not None: os.remove(self.model_path)

class maxentClassifier(Classifier):
  __name__ = "maxent"

  def __init__(self, model_path, num_classes, name=None):
    self.toolpath = config.getpath('tools','maxent')
    if name:
      self.__name__ = name
    Classifier.__init__(self)
    self.model_path  = model_path
    self.num_classes = num_classes
    self.clear_temp  = config.getboolean('debug', 'clear_temp_files')
  
  def _classify(self, feature_map):
    test  = tempfile.NamedTemporaryFile(delete=self.clear_temp)

    # create and write the test file
    self.logger.debug("writing test file: %s", test.name)
    SVMFileWriter.write(test, feature_map)
    test.flush()

    num_test_docs = feature_map.shape[0]

    # create a temporary file for the results
    result_file, result_path = tempfile.mkstemp()
    os.close(result_file)

    # call the classifier
    classif_cmd = [ self.toolpath, '-p', '-m', self.model_path, '--detail', 
        '-o', result_path, test.name ]
    self.logger.debug("Classifying maxent: %s", ' '.join(classif_cmd))
    retcode = subprocess.call(classif_cmd, stdout=open('/dev/null'))
    if retcode:
      self.logger.critical("Classifying maxent failed")
      raise ValueError, "Classif maxent returned %s"%(str(retcode))

    result_file = open(result_path)
    classifications = numpy.zeros((num_test_docs, self.num_classes), dtype='float')

    for i,line in enumerate(result_file):
      terms = line.split()
      while terms != []:
        # Read pairs of outcome, probability
        # TODO: Maxent handles multiclass by computing the joint probability.
        #       We hackishly assign the same probability to both classes.
        #       This will cause problems if we interpret as singlehighest 
        #         - it makes us worse than we really are
        #         - it really is an interpretation issue
        outcomes = map(int,terms.pop(0).split(','))
        probability = float(terms.pop(0))
        for outcome in outcomes:
          classifications[i, outcome] = probability

    # Dispose of the unneeded output file
    result_file.close()
    if self.clear_temp:
      os.remove(result_path)

    return classifications

