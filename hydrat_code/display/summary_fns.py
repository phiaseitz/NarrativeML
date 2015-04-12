from hydrat.result import CombinedMacroAverage, CombinedMicroAverage, PRF
import numpy

"""
This module provides an assortment of functions for common tasks in generating
'summaries' of results. The basic model here is that for each taskset we run, we
get a number of raw classifier output. We need to process this into meaningful
data points, such as precision/recall/fscore, and we also want to get some
statistics such as the number of featuresets used in the experiment. To do this,
our overall framework is as follows:

We implement a function that takes a :class:`hydrat.result.Result`, and returns
a dictionary with interesting values. For example, we may return something as simple
as {'precision':0.5, 'recall':0.5}. 

For more sophisticated processing, we can use AggregateSummaryFunction, which takes 
a list of functions such as the one described above, and applies them to the result
in order, unioning the output of each and returning the final result. For example,
if we had a function which returned the precision, and a function that returned
the recall, we would use AggregateSummaryFunction to build a function that returns 
both precision and recall.

AggregateSummaryFunction takes a second list of functions, 'transformers'. These
functions are applied to the unioned summary dictionary after it is built.
These functions can perform roles such as preformatting output (e.g. converting
all floats to 3 significant figure string representations).

This extensible model allows us to build a custom analysis according to our needs
by composing analytical components.
"""
class AggregateSummaryFunction(object):
  def __init__(self, extractors = None, transformers = None):
    self.extractors = extractors if extractors is not None else []
    self.transformers = transformers if transformers is not None else []

  def __call__(self, result, interpreter):
    summary = dict()
    for e in self.extractors:
      summary.update(e(result, interpreter))
    for t in self.transformers:
      summary = t(summary)
    #import pdb;pdb.set_trace()
    return summary 

def is_probabilistic(matrix):
  """
  Approximate hack to figure out if a classification output is probabilistic.
  The condition is basically that all all values in the matrix must be between 
  0.0 and 1.0.
  """
  if not issubclass(matrix.dtype.type, numpy.float):
    return False
  if numpy.logical_or(matrix < 0.0, matrix > 1.0).any():
    return False
  return numpy.logical_and(0.0 < matrix, matrix < 1.0).any()
  
def preformat_floats(entry):
  """
  Convert all floats to 4-decimal-place string representations
  """
  for e in entry:
    if isinstance(entry[e], float):
      entry[e] = "%.4f" % entry[e]
  return entry

def featureset_count(result, interpreter):
  """
  Counts the number of featuresets used in the result.
  """
  if 'feature_desc' in result.metadata:
    nf = len(result.metadata['feature_desc'])
  else:
    nf = 0
  return dict(num_featuresets = nf)

def expand_features(result, interpreter):
  """
  Creates one boolean entry for each feature present
  """
  if 'feature_desc' not in result.metadata: return {}
  e = result.metadata['feature_desc'] 
  if type(e[0]) == str:
    fs = set([ e[0] ])
  else:
    fs = set( f[0] for f in result.metadata['feature_desc'] )
  result = dict()
  for f_name in fs:
    result['feat_' + f_name] = True
  return result

def replace_ordinal_learner(result, interpreter):
  """
  Expand the learner_desc for ordinal learners, adding an
  'isOrdinal' key and an 'ordering' key, and replacing the 'learner' and 'learner_params'
  keys with the inner learner.
  """
  entry = {'isOrdinal':False}
  if result.metadata['learner'] == 'ordinal':
    entry['ordering'] = result.metadata['learner_params']['ordering']
    entry['learner'], entry['learner_params'] = result.metadata['learner_params']['learner']
    entry['isOrdinal'] = True
  return entry
  
def micro_scores(result, interpreter):
  """
  Calculate microaveraged PRF across all classes across all folds
  """
  micro_score = CombinedMicroAverage(result.overall_confusion_matrix(interpreter))
  i_p = micro_score.precision
  i_r = micro_score.recall
  i_f = micro_score.fscore
  entry =\
    { 'micro_precision'  : i_p
    , 'micro_recall'     : i_r
    , 'micro_fscore'     : i_f
    }
  return entry

def macro_scores(result, interpreter):
  """
  Calculate macroaveraged PRF across all classes across all folds
  """
  macro_score = CombinedMacroAverage(result.overall_confusion_matrix(interpreter))
  a_p = macro_score.precision
  a_r = macro_score.recall
  a_f = macro_score.fscore
  entry =\
    { 'macro_precision'  : a_p
    , 'macro_recall'     : a_r
    , 'macro_fscore'     : a_f
    }
  return entry

def result_metadata(result, interpreter):
  """
  Propagate metadata already present in the result object. This will cover things
  such as the identity of the learner, feature set and class set.
  Also adds keys to reflect the identity of the interpreter, whether the output is
  probabilistic, and a 'link' which is used to link to a detailed per-result 
  analysis.
  """
  entry =\
    { 'interpreter'      : interpreter.__name__
    , 'probabilisitc'    : is_probabilistic(result.results[0].classifications)
    }
  entry.update(result.metadata)
  #TODO: This should go elsewhere
  entry['link'] = "<a href=%s>link</a>" % (str(entry['uuid'])+'.html')
  return entry

class RMSError(object):
  """
  Compute the RMS error of a result, with respect to a certain ordering
  on classes. We need the parent datastore in order to resolve said ordering.
  """
  def __init__(self, data):
    self.data = data

  def __call__(self, result, interpreter):
    space_name = result.metadata['class_space']
    space = self.data.get_Space(space_name)
    ordering = numpy.array(space).argsort()
    matrix = result.overall_classification_matrix(interpreter)
    assert matrix.shape[0] == matrix.shape[1] == len(ordering)
    errors = []
    for i in range(len(ordering)):
      for j in range(len(ordering)):
        # Calculate the squared error from each classification pair
        errors.extend(matrix[i,j] * [(i - j) ** 2])
    # Calculate the root of the mean
    err = numpy.sqrt(numpy.mean(errors))
    return dict(ordering=list(ordering), rms_error=err)

def mean_absolute_error(matrix, ordering):
  """
  Compute the mean absolute error given a classification matrix, and an
  ordering between classes. Assumes unit distance between classes.
  """
  assert matrix.shape[0] == matrix.shape[1] == len(ordering)
  errors = []
  for i in range(len(ordering)):
    for j in range(len(ordering)):
      # Calculate the absolute error from each classification pair
      # Absolute difference between the ordering, multiplied
      # but the number of times this mistake was made.
      errors.extend([abs(ordering[i] - ordering[j])] * matrix[i,j])
  # Calculate the mean
  err = numpy.mean(errors)
  return err

class MAEError(object):
  """
  Compute the MAE error of a result, with respect to a certain ordering
  on classes. We need the parent datastore in order to resolve said ordering.
  """
  def __init__(self, data):
    self.data = data

  def __call__(self, result, interpreter):
    space_name = result.metadata['class_space']
    space = self.data.get_Space(space_name)
    ordering = numpy.array(space).argsort()
    matrix = result.overall_classification_matrix(interpreter)
    err = mean_absolute_error(matrix, ordering)
    return dict(ordering=list(ordering), mae_error=err)


class PerfoldConfusionMetric:
  """
  Calculate a confusion matrix metric on a per-fold basis
  """
  def __init__(self, key, aggregator, metric):
    self.aggregator = aggregator
    self.metric = metric
    self.key = key

  def __call__(self, result, interpreter):
    fold_results = {}
    for r in result.results:
      index = r.metadata['index']
      cm = r.confusion_matrix(interpreter)
      fold_results[index] = self.aggregator(cm, self.metric)
    return { 'perfold_'+self.key : fold_results} 

class PerfoldClassificationMetric:
  """
  Calculate a classification matrix metric on a per-fold basis
  """
  def __init__(self, data, key, metric):
    self.key = key
    self.metric = metric
    self.data = data

  def __call__(self, result, interpreter):
    fold_results = {}
    for r in result.results:
      space_name = result.metadata['class_space']
      space = self.data.get_Space(space_name)
      #TODO ranklist might be what we actually want!
      ordering = numpy.array(space).argsort()
      index = r.metadata['index']
      cm = r.classification_matrix(interpreter)
      fold_results[index] = self.metric(cm, ordering) 
    return { 'perfold_'+self.key : fold_results} 
   
# Some predefined basic AggregateSummaryFunction
sf_basic =  AggregateSummaryFunction\
  ( [ macro_scores
    , micro_scores
    , result_metadata
    ]
  )
 
sf_featuresets =  AggregateSummaryFunction\
  ( [ macro_scores
    , micro_scores
    , result_metadata
    , featureset_count 
    , expand_features
    ]
  )

sf1 = AggregateSummaryFunction( [ macro_scores
                                , micro_scores
                                , result_metadata
                                , featureset_count 
                                ]
                              , [ preformat_floats
                                ]
                              )
