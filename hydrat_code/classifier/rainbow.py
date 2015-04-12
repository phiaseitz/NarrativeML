from abstract import NullLearner, LearnerlessClassifier 
from hydrat import config
import tempfile
import shutil
import os
import numpy
from scipy.sparse import issparse

from hydrat.classifier.meta.weighted import WeightedLearner
from hydrat.common.transform.weight import Discretize

# NOTE:
# Rainbow upstream is pretty much dead McCallum is working on Mallet now, last release was
# 2002 and ubuntu has pulled it from repos. This wrapper will not be developed any further.

#__all__ = [ "naivebayesL", "prindL", "tfidfL" ]
__all__ = [ "naivebayesL", "naivebayesdiscreteL", "tfidfL" ] #TODO: Find out why prind crashes
class RainbowFileWriter(object):
  @staticmethod
  def instance(docid, fv, cv = None):
    output = ""
    if cv is None:
      # Label all unseen documents as class 0
      classids = numpy.array([0]) 
    else:
      classids = numpy.arange(len(cv))[cv]
    for classid in classids:
      output += "%s %s" % (docid, classid)
      fv.sort_indices()
      # Looks like this is not a problem after all.
      #if len(fv.data) == 0:
      #  raise ValueError, "Rainbow cannot handle featureless vectors!"
      if (fv.data < 0).any():
        raise ValueError, "Rainbow cannot handle negative features!"
      for feature, value in zip(fv.indices, fv.data):
        if type(value) == float:
          raise ValueError, "Rainbow cannot handle continuous-valued features" 
        output += "  %s %d" % ("feat" + str(feature), value)
      output += "\n"
    return output

  @staticmethod
  def writefile(file, tag, feature_map, class_map = None):
    if class_map is not None:
      assert feature_map.shape[0] == class_map.shape[0]
    for i in xrange(feature_map.shape[0]):
      d = "%s_%d" % (tag, i)
      if class_map is None:
        instance = RainbowFileWriter.instance(d ,feature_map[i])
      else:
        instance = RainbowFileWriter.instance(d, feature_map[i], class_map[i])
      file.write(instance)

# This classifier ends up a little bit funny because we are
# by and large usurping rainbow's functionality. Rainbow
# was designed to parse features itself, using a BOW model.
# We are instead injecting features into it directly, by
# simulating a BOW model out of whatever model we have used.
# One problem with this is that it becomes harder to do testing 
# of unseen data, as we build a model with both our training and
# test data in it.
class RainbowClassifier(LearnerlessClassifier):
  __name__    = "rainbow"
  _train_tag  = "train"
  _test_tag   = "test"

  def __init__(self, name, feature_map, class_map, method):
    assert issubclass(feature_map.dtype.type, numpy.integer)
    LearnerlessClassifier.__init__(self, name, feature_map, class_map)
    self.method      = method
    self.rainbowexe  = config.get('tools', 'rainbow') 

  def _rc(self,command):
    full_command = self.rainbowexe + command
    self.logger.debug("calling rainbow: %s",full_command)
    cmd = os.popen(full_command)
    cmd_out = cmd.read()
    returncode = cmd.close()
    if returncode is not None:
      self.logger.critical( cmd_out )
      raise ValueError, "Rainbow returned %d" % (returncode)
    return cmd_out

  def _parseClassifications(self, output, num_docs, num_classes):
    classifications = numpy.zeros((num_docs, num_classes), dtype = 'float')
    for line in output.split('\n'):
      if line.startswith('#'): continue
      tokens = line.split()
      if tokens == []: break # No more output
      assert tokens[0].startswith(self._test_tag)
      doc_index = int(tokens[0].rsplit('_').pop())
      trueclass = int(tokens[1])
      for token in tokens[2:]:
        klass,value = token.split(":")
        classifications[doc_index, int(klass)] = float(value)
    return classifications 

  def _classify(self, test_fv):
    writer = RainbowFileWriter

    #Create and write the data file
    data = tempfile.NamedTemporaryFile()
    writer.writefile(data, self._train_tag, self.train_fv, self.train_cv )
    writer.writefile(data, self._test_tag, test_fv ) 
    data.flush()

    #Create a directory for the model
    model_path = tempfile.mkdtemp()

    self.logger.info("parsing model")
    self._rc( " -v 0 -d %s --index-matrix=siw %s" 
            % (model_path, data.name) 
            )

    #Create a file to store test ids
    testid_path = tempfile.NamedTemporaryFile()
    testid_path.write("\n".join(    "%s_%d" % (self._test_tag,id)
                               for  id 
                               in   xrange(test_fv.shape[0])
                               )
                     )
    testid_path.flush()

    self.logger.info("classifying")
    output = self._rc( " -v 0 -d %s --test-set=%s --method=%s --test=1" 
                     % (model_path, testid_path.name, self.method) 
                     )
    num_docs     = test_fv.shape[0]
    num_classes  = self.train_cv.shape[1]

    # Clean up temporary files
    shutil.rmtree(model_path)
    return self._parseClassifications(output, num_docs, num_classes)

def naivebayesL():
  return NullLearner(RainbowClassifier, "rainbow_nb", "naivebayes")

def naivebayesdiscreteL():
  return WeightedLearner(naivebayesL(), Discretize())

def prindL():
  return NullLearner(RainbowClassifier, "rainbow_prind", "prind")

def tfidfL():
  return NullLearner(RainbowClassifier, "rainbow_tfidf", "tfidf")

