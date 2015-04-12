"""
hydrat classifier interface to Vowpal Wabbit

https://github.com/JohnLangford/vowpal_wabbit/wiki
"""

import tempfile
import os
import subprocess
import numpy as np
import csv
import logging

from itertools import izip
from contextlib import closing

from hydrat import config
from hydrat.configuration import Configurable, EXE
from hydrat.classifier.abstract import Learner, Classifier, NotInstalledError
from hydrat.task.sampler import isOneofM
from hydrat.common.timer import Timer

logger = logging.getLogger(__name__)

tempfile.tempdir = config.getpath('paths','scratch')
debug_quiet = config.getboolean('debug', 'quiet_extern')

def vw_instance(fv, class_label=None):
  # TODO: We don't implement importance, tag and namespace for now.
  #
  # The raw (plain text) input data for VW should have one example per line. 
  # Each example should be formatted as follows.
  #
  # [Label] [Importance [Tag]]|Namespace Features |Namespace Features ... |Namespace Features
  #   where
  # Namespace=String[:Value]
  #   and
  # Features=(String[:Value] )*
  # TODO: Do we need to sort these indices? indices and data should be aligned...
  f_iter = izip(fv.indices,fv.data)
  features = ''.join("{0}:{1} ".format(*i) for i in f_iter)
  retval = '{0} | {1}\n'.format(class_label if class_label else '',features) 
  return retval
  
  
def write_vw_file(output, feature_map, class_map=None):
  """
  @param output file-like object to write output to
  @param feature_map 2D csr_matrix [instances, features]
  @param class_map 2D ndarray (boolean) [instances, classes]
  """

  logger.debug('write_vw_file: fm:{0}'.format(feature_map.shape))
  if class_map is None:
    # This is a test file, all instances will be labelled as class 0
    class_labels = np.zeros((feature_map.shape[0],), dtype=int)
  else:
    # This is a training file
    if feature_map.shape[0] != class_map.shape[0]:
      raise ValueError("instance count mismatch")

    if not isOneofM(class_map):
      raise ValueError("vw only supports mono-label instances")
    class_labels = np.argmax(class_map, axis=1) + 1
    
  with Timer() as t:
    for i, fv in enumerate(feature_map):
      # offset by 1 as 0 is for unlabeled
      class_label = class_labels[i]
      output.write(vw_instance(fv, class_label))
      if not debug_quiet and i and i % 1000 == 0:
        # NOTE: The actual number i is off-by-one, but since this is mostly to give a rate
        #       indication its not critical.
        print 'wrote {0} instances in {1}s ({2} i/s)'.format(i, t.elapsed, t.rate(i))
    logger.debug('wrote {0} instances in {1}s ({2} i/s)'.format(i+1, t.elapsed, t.rate(i+1)))

  output.flush()





class vwL(Configurable, Learner):
  __name__ = "vowpalwabbit"
  requires= {
    ('tools', 'vw') : EXE('vw'),
  }

  def __init__(self, **kwargs):
    self.kwargs = kwargs
    self.vwpath = config.getpath('tools','vw')
    if kwargs:
      raise NotImplementedError("not handling any kwargs for vw yet")

    Learner.__init__(self)

  def is_pickleable(self):
    return True 

  def __getstate__(self):
    return self.kwargs

  def __setstate__(self, value):
    self.__init__(**value)

  def _check_installed(self):
    if not os.path.isfile(self.vwpath):
      raise NotInstalledError

  def _params(self):
    return self.kwargs

  def _learn(self, feature_map, class_map):
    num_class = class_map.shape[1]

    # Write training file
    with closing(tempfile.NamedTemporaryFile(suffix='.vw',delete=False)) as t_file:
      self.logger.debug('writing train file: {0}'.format(t_file.name))
      write_vw_file(t_file, feature_map, class_map)

    # Generate model path
    m_file, m_path = tempfile.mkstemp(suffix='.vwmodel')
    os.close(m_file)

    training_cmd = [ self.vwpath ]
    training_cmd += [ '--loss_function', 'logistic', ]
    if num_class > 2:
      # More than 2 classes, we handle this with oaa for now
      training_cmd += [ '--oaa', str(num_class) ]
    training_cmd += [ '-d', t_file.name ]
    training_cmd += [ '-f', m_path ]
    if debug_quiet:
      training_cmd += [ '--quiet' ]

    # Call vw to train
    self.logger.debug('calling vw: {0}'.format(' '.join(training_cmd)))
    retcode = subprocess.call(training_cmd)
    if retcode:
      raise ValueError("training vw returned {0}".format(retcode))

    return vwC(m_path, num_class, self.kwargs)
      

class vwC(Classifier):
  __name__ = "vowpalwabbit"
  requires= {
    ('tools', 'vw') : EXE('vw'),
  }

  def __init__(self, m_path, num_class, kwargs):
    Classifier.__init__(self)
    self.m_path = m_path
    self.num_class = num_class
    self.kwargs = kwargs
    self.vwpath = config.getpath('tools','vw')
    self.clear_temp = config.getboolean('debug','clear_temp_files')

  def is_pickleable(self):
    return False

  def __del__(self):
    if self.clear_temp:
      if self.m_path is not None:
        os.remove(self.m_path)

  def _classify(self, feature_map):
    # Get a path for predictions
    p_file, p_path = tempfile.mkstemp()
    os.close(p_file)

    with closing(tempfile.NamedTemporaryFile(suffix='.vw',delete=self.clear_temp)) as t_file:
      self.logger.debug('writing test file: {0}'.format(t_file.name))
      write_vw_file(t_file, feature_map)

      test_cmd = [ self.vwpath ]
      test_cmd += [ '--loss_function', 'logistic', ]
      if self.num_class > 2:
        # More than 2 classes, we handle this with oaa for now
        test_cmd += [ '--oaa', str(self.num_class) ]
      test_cmd += [ '-i', self.m_path ]
      test_cmd += [ '-t' ]
      test_cmd += [ '-d', t_file.name ]
      test_cmd += [ '-r', p_path ]
      if debug_quiet:
        test_cmd += [ '--quiet' ]

      # Call vw to classify
      self.logger.debug('calling vw: {0}'.format(' '.join(test_cmd)))
      retcode = subprocess.call(test_cmd)
      if retcode:
        raise ValueError("training vw returned {0}".format(retcode))

    # process the predictions
    preds = []
    with open(p_path) as p_file:
      # based on the approach of 
      # http://fastml.com/predicting-closed-questions-on-stack-overflow/
      reader = csv.reader(p_file, delimiter=' ')
      for row in reader:
        # discard every second row, it is the instance tag which
        # we do not use.
        reader.next() 
        
        preds.append( [ float(e.split(':')[1]) for e in row ] )
    # Apply sigmoid function
    preds = 1 / (1 + np.exp(-np.array(preds)))
    # Renormalize
    retval = preds / preds.sum(axis=1)[:,None]

    assert feature_map.shape[0] == retval.shape[0]
    assert retval.shape[1] == self.num_class

    if self.clear_temp and os.path.isfile(p_path):
      os.remove(p_path)

    return retval

