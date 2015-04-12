from hydrat.classifier.abstract import Learner, Classifier
from itertools import izip
from hydrat import config
from hydrat.configuration import Configurable, EXE
from hydrat.common.sequence import matrix2sequence

import tempfile
import os
import logging
import time
import numpy
import sys

def parse_crfsgd_output(data):
  return [ line[-1] for line in data.split('\n') if len(line) > 0 ]

class CRFFileWriter(object):

  @staticmethod
  def instance(fv, cv = None):
    fv = fv.toarray()
    
    if cv is not None:
      classids = numpy.arange(cv.shape[0])[cv]
    else:
      classids = [""] 
    classlabelblock = ",".join(str(id) for id in classids)
    features = " ".join(    str(fv[0,i]) 
                       for  i
                       in   xrange(len(fv[0])) 
                       )
    return "%s %s\n" % (features, classlabelblock)

  @staticmethod
  def writefile(file, fvs, sequence, cvs = None):
    """ Write a CRFSGD compatible file. Note that sequence is 
    expected to be in list-of-lists format.
    """
    if cvs is not None: assert fvs.shape[0] == cvs.shape[0]
    for thread in sequence:
      for i in thread:
        if cvs is not None:
          one_line = CRFFileWriter.instance(fvs[i], cvs[i])
        else:
          one_line = CRFFileWriter.instance(fvs[i])
        file.write(one_line)
      file.write('\n')

class crfsgdL(Configurable, Learner):
  __name__ = 'crfsgd'
  requires =\
    { ('tools','crfsgd')            : EXE('crfsgd')
    , ('tools','crfsgd-conlleval')  : EXE('conlleval')
    }

  def __init__(self, capacity=1.0):
    self.clear_temp = config.getboolean('debug', 'clear_temp_files')
    self.toolpath = config.getpath('tools', 'crfsgd')
    self.capacity = capacity
    Learner.__init__(self)

    self.model_path = None

  def __del__(self):
    if self.clear_temp:
      if self.model_path is not None: os.remove(self.model_path)

  def _params(self):
    return dict(capacity=self.capacity)
    
  def _learn(self, feature_map, class_map, sequence):
    seq = matrix2sequence(sequence)
    #build the template for CRF learner
    #TODO: note that I did not put *identifiers* at the moment 
    template_len = feature_map.shape[1]
    template = tempfile.NamedTemporaryFile(delete=self.clear_temp)
    template.write("# Unigram\n")
    self.logger.debug("writing template file: %s", template.name)
    for i in range(template_len):
      template.write("U"+str(i)+":%x[0,"+str(i)+"]\n")
    template.flush()

     #Create and write the training file
    train = tempfile.NamedTemporaryFile(delete=self.clear_temp)
    self.logger.debug("writing training file: %s", train.name)
    CRFFileWriter.writefile(train, feature_map, seq, class_map)
    train.flush()

    #Create a temporary file for the model
    model_file, self.model_path = tempfile.mkstemp()
    self.logger.debug("model path: %s", self.model_path)
    os.close(model_file)

    training_command =\
      "%s -e %s -q -c %f %s %s %s" % ( self.toolpath 
                      , config.getpath('tools', 'crfsgd-conlleval')
                      , self.capacity
                      , self.model_path
                      , template.name
                      , train.name
                      )
    self.logger.debug("Training CRF: %s", training_command)
    # Try to replace os.popen with subprocess (http://docs.python.org/library/subprocess.html)
    # Alternatively, if there is progress output from 'crfsgd', use Pexpect (http://www.noah.org/wiki/Pexpect)
    process = os.popen(training_command)
    output = process.read()
    return_value = process.close()
    if return_value:
      self.logger.critical("Training 'crfsgd' failed with output:")
      self.logger.critical(output)
      raise ValueError, "Training 'crfsgd' returned %s"%(str(return_value))
    return crfsgdC( self.model_path
                  , self.toolpath
                  , class_map.shape[1]
                  )

class crfsgdC(Classifier):
  __name__ = "crf"

  def __init__(self, model_path, toolpath, num_classes):
    Classifier.__init__(self)
    self.model_path  = model_path
    self.toolpath  = toolpath 
    self.num_classes = num_classes
    self.clear_temp  = config.getboolean('debug','clear_temp_files')

  def _classify(self, feature_map, sequence):
    seq = matrix2sequence(sequence)
    #Create and write the test file
    test  = tempfile.NamedTemporaryFile(delete=self.clear_temp)
    self.logger.debug("writing test file: %s", test.name)
    CRFFileWriter.writefile(test, feature_map, seq)
    test.flush()

    # Do the classification
    result_file, result_path = tempfile.mkstemp()
    os.close(result_file)
    classif_command = "%s -t %s %s %s" % ( self.toolpath 
                                      , self.model_path
                                      , test.name
                                      , ">"+result_path
                                      )
    self.logger.debug("Classifying CRF: %s", classif_command)
    process = os.popen(classif_command)
    output = process.read()
    return_value = process.close()
    if return_value:
      self.logger.critical("Classifying 'crfsgd' failed with output:\n"+output)
      raise ValueError, "Classif 'crfsgd' returned %s"%(str(return_value))

    # Parse CRFSGD output
    with open(result_path) as f:
      classifications = parse_crfsgd_output(f.read())

    num_test_docs = feature_map.shape[0]
    cm = numpy.zeros((feature_map.shape[0], self.num_classes), dtype='bool')

    post_order = numpy.hstack(seq)
    assert len(post_order) == len(classifications)
    for instance, cl in zip(post_order, classifications):
      cm[instance,cl] = True

    # Cleanup
    if self.clear_temp:
      os.remove(result_path)
    return cm 


