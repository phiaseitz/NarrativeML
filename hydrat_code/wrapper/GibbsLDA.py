"""
Python wrapper for GibbsLDA++
http://gibbslda.sourceforge.net/
"""
import subprocess
import tempfile
import os
import shutil
import pexpect
import re
import numpy
import scipy.sparse
import logging

# TODO: Support load/save of model
#       Support access to various mappings provided by the model

logger = logging.getLogger(__name__)


LDA_EXE = ''
TMP = './scratch'

RE_ITERATION = re.compile(r'Iteration (?P<count>\d+) ...')

enc_symbols = []
for s in ['09', 'az','AZ']:
  for o in range(ord(s[0]), ord(s[1])+1):
    enc_symbols.append(chr(o))
enc_symbols = ''.join(enc_symbols)

def encode(feature):
  retval = []
  symbol_count = len(enc_symbols)
  while feature:
    retval.append(enc_symbols[feature % symbol_count])
    feature /= symbol_count
  return ''.join(reversed(retval))

def array2file(array, fileobj):
  """
  Produce an input file for GibbsLDA from a sparse array.
  For each row, for each index, we produce a number of tokens
  according to the value at that index.
  The actual token produced is not important, as long as we are
  able to recover the original index from it.
  """
  fileobj.write(str(array.shape[0]) + '\n')
  for row in array:
    instance = []
    for feature in row.nonzero()[1]:
      count = int(row[0,feature])
      instance.extend([encode(feature)] * count)
    fileobj.write(' '.join(instance) + '\n')

class GibbsLDA(object):
  def __init__( self
              , alpha = None
              , beta = 0.1
              , ntopics = 100
              , niters = 2000
              , infiters = 30
              , savestep = None
              , twords = 0
              , exe = LDA_EXE
              , tmp = TMP
              , clear_temp = True
              , timeout = 60
              ):
    if alpha is None:
      self.alpha = 50 / ntopics
    else:
      self.alpha = alpha
    self.beta = beta
    self.ntopics = ntopics
    self.niters = niters
    self.infiters = infiters
    self.savestep = savestep if savestep is not None else 0
    self.twords = twords
    self.clear_temp = clear_temp
    self.timeout = timeout
    self.exe = exe
    self.tmp = tmp
    self.workdir = os.path.abspath(tempfile.mkdtemp(prefix='GibbsLDA',dir=self.tmp))
    self.trained = False
    if not (os.path.exists(exe) and os.access(exe, os.X_OK)):
      raise ValueError, "'%s' is not a valid executable" % exe
 
  def __del__(self):
    if self.clear_temp and os.path.exists(self.workdir):
      shutil.rmtree(self.workdir)

  def estimate(self, feature_map, progress_callback = None):
    """
    <model_name>.phi: 
      This file contains the word-topic distributions, i.e., p(wordw|topict). 
      Each line is a topic, each column is a word in the vocabulary
    <model_name>.theta: 
      This file contains the topic-document distributions, i.e., p(topict|documentm). 
      Each line is a document and each column is a topic.
    <model_name>.tassign: 
      This file contains the topic assignment for words in training data. 
      Each line is a document that consists of a list of <wordij>:<topic of wordij>
      Could use this as a token stream!
    """
    with tempfile.NamedTemporaryFile\
            ( prefix='GibbsLDA-'
            , suffix='-learn'
            , dir=self.workdir
            , delete=self.clear_temp
            ) as f:
      # Write the training file
      logger.debug('writing training file: "%s"', f.name)
      array2file(feature_map, f)
      command =\
        [ self.exe
        , '-est'
        , '-alpha', self.alpha
        , '-beta', self.beta
        , '-ntopics', self.ntopics
        , '-niters', self.niters
        , '-savestep', self.savestep
        , '-twords', self.twords
        , '-dfile', f.name
        ]
      command = ' '.join(map(str, command))
      logger.debug(command)
      lda_instance = pexpect.spawn(command)
      lda_instance.expect(r'Sampling (?P<count>\d+) iterations!')
      niters = int(lda_instance.match.group('count'))

      # Monitor output of GibbsLDA to account for progress
      for i in range(niters):
        # TODO: loosen timeout?
        lda_instance.expect(RE_ITERATION, timeout=self.timeout)
        if progress_callback is not None:
          progress_callback(i+1)

      lda_instance.expect(r'Saving the final model!')
      lda_instance.expect(pexpect.EOF)
    self.trained = True

  @property
  def model(self):
    """
    Produce an object representing the current model
    """
    if not self.trained:
      raise ValueError, "Not trained"
    paths=[
      "model-final.others",
      "model-final.phi",
      "model-final.theta",
      "model-final.tassign",
      "model-final.twords",
      "wordmap.txt",
      ]
    retval = {}
    for path in paths:
      full_path = os.path.join(self.workdir, path)
      if os.path.exists(full_path):
        with open(full_path) as f:
          retval[path] = f.read()
    return retval

  def load_model(self, model):
    """
    Load a previously saved model
    """
    for path in model:
      with open(os.path.join(self.workdir, path), 'w') as f:
        f.write(model[path])
    self.trained = True

  @property
  def topics(self):
    if not self.trained:
      raise ValueError, "Not trained"
    theta = numpy.genfromtxt(os.path.join(self.workdir,'model-final.theta'))
    theta = scipy.sparse.csr_matrix(theta)
    return theta

  def continue_estimate(self):
    raise NotImplementedError, "Continued estimation not yet implemented"
    

  def apply(self, feature_map, progress_callback=None):
    with tempfile.NamedTemporaryFile\
            ( prefix='GibbsLDA-'
            , suffix='-apply'
            , dir=self.workdir
            , delete=self.clear_temp
            ) as f:
      array2file(feature_map, f)
      command =\
        [ self.exe
        , '-inf'
        , '-dir', self.workdir
        , '-model', 'model-final'
        , '-niters', self.infiters
        , '-twords', self.twords
        , '-dfile', os.path.basename(f.name)
        ]
      command = ' '.join(map(str, command))
      logger.debug(command)

      lda_instance = pexpect.spawn(command)
      lda_instance.expect(r'Sampling (?P<count>\d+) iterations for inference!')
      niters = int(lda_instance.match.group('count'))

      for i in range(niters):
        lda_instance.expect(RE_ITERATION)
        if progress_callback is not None:
          progress_callback(i+1)

      lda_instance.expect(r'Saving the inference outputs!')
      lda_instance.expect(pexpect.EOF)
      theta = numpy.genfromtxt(f.name+'.theta')
    theta = scipy.sparse.csr_matrix(theta)
    return theta

    
