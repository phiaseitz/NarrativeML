import logging
import tempfile
import os
import numpy

from hydrat.preprocessor.model.arff import arff_export
from hydrat.preprocessor.model import ImmediateModel
from hydrat.classifier.common import run_command
from hydrat.classifier.abstract import Learner, Classifier
from hydrat import config
from hydrat.configuration import is_exe, EXE, Configurable, FILE

"""
Wrapper for WEKA

Mark Hall, Eibe Frank, Geoffrey Holmes, Bernhard Pfahringer, Peter Reutemann, 
Ian H. Witten (2009); The WEKA Data Mining Software: An Update; 
SIGKDD Explorations, Volume 11, Issue 1.

http://www.cs.waikato.ac.nz/ml/weka/

..todo: Check that the interfacing is done correctly. Results are surprisingly low at times.
"""

class WekaLearner(Configurable, Learner):
  __name__ = 'weka'
  requires=\
    { ('tools','java-bin') : EXE('java')
    , ('tools','weka')     : FILE('weka.jar', paths=['/usr/share/java'])
    }

  def __init__(self, cl_name, options = ""):
    Learner.__init__(self)
    self.cl_name = cl_name
    self.options  = options

  def _check_installed(self):
    if not is_exe(config.get('tools','java-bin')):
      raise ValueError, "java not installed"
    if not os.path.exists(config.get('tools','weka')):
      raise ValueError, "weka.jar not found at %s" % config.get('tools','weka')


  def _params(self):
    return {'cl_name':self.cl_name, 'options':self.options}

  def _learn(self, feature_map, class_map):
    model = ImmediateModel(feature_map, class_map)

    train_file = tempfile.NamedTemporaryFile(suffix='.arff')

    arff_export(train_file, model)
    train_file.flush()
    self.logger.debug('train path: %s', train_file.name)

    model_file, model_path = tempfile.mkstemp(suffix='.weka_model')
    os.close(model_file)
    self.logger.debug("model path: %s", model_path)

    weka_command = " ".join(( config.get('tools','java-bin')
                            , "-cp", config.get('tools','weka')
                            , "weka.classifiers." + self.cl_name
                            , self.options
                            , "-t", train_file.name
                            , "-d", model_path
                            , "-c", "1" # Class label is first attribute
                           ))
    self.logger.debug("Calling Weka: %s", weka_command)
    run_command(weka_command)

    self.logger.debug("Returning Classifier")
    return WekaClassifier(self.cl_name, model_path, model.classlabels)

    
class WekaClassifier(Classifier):
  __name__ = "weka"
  def __init__(self, cl_name, model_path, classlabels ):
    self.__name__ = 'weka_' + cl_name
    Classifier.__init__(self)
    self.cl_name = cl_name
    self.model_path = model_path
    self.classlabels = classlabels

  def __del__(self):
    pass
    #os.remove(self.model_path)

  def _classify(self, feature_map):
    model = ImmediateModel(feature_map, classlabels = self.classlabels)

    test_file = tempfile.NamedTemporaryFile(suffix='.arff')

    arff_export(test_file, model)
    test_file.flush()
    self.logger.debug('test path: %s', test_file.name)

    weka_command = " ".join(( config.get('tools','java-bin')
                            , "-cp", config.get('tools','weka')
                            , "weka.classifiers." + self.cl_name
                            , "-l", self.model_path
                            , "-T", test_file.name
                            , "-c 1" # Class label is first attribute
                            , "-p 0"
                            , "-distribution"
                           ))
    self.logger.debug("Calling Weka: %s", weka_command)
    output = run_command(weka_command)
    class_map = numpy.empty((feature_map.shape[0], len(self.classlabels)), dtype=float)

    for line in output.split('\n'):
      try:
        (id, gs, cl, err, dist) = line.split()
        instance_id = int(id) - 1
        # class_id = int(cl.split(':')[0]) - 1
      except ValueError:
        continue

      for class_index, value_str in enumerate(dist.split(',')):
        if value_str[0] == '*':
          value = float(value_str[1:])
        else:
          value = float(value_str)
        class_map[instance_id, class_index] = value
       
    return class_map

def nbL():
  return WekaLearner('bayes.NaiveBayes')

def bayesnetL():
  return WekaLearner('bayes.BayesNet')

def perceptronL():
  return WekaLearner('functions.MultilayerPerceptron')

def baggingL():
  return WekaLearner('meta.Bagging')

def stackingL():
  return WekaLearner('meta.Stacking')

def j48L():
  return WekaLearner('trees.J48')

def majorityclassL():
  return WekaLearner('rules.ZeroR')
