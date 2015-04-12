import numpy
import hydrat
import hydrat.common.markup as markup
from hydrat.result import CombinedMacroAverage, CombinedMicroAverage
from hydrat.result import Microaverage, Macroaverage, PRF
from hydrat.display.sparklines import boxplot

class Summary(object):
  def __init__(self):
    self.result = None
    self.interpreter = None
    self.handlers = {}

  def init(self, result, interpreter):
    """
    Override this if you need to pre-compute some values if
    a result and/or interpreter change.
    """
    # TODO: How to ensure any extended handlers get a proper init if we override this??
    self.result = result
    self.interpreter = interpreter

  def __call__(self, result, interpreter):
    self.init(result, interpreter)
    return dict( (key, self[key]) for key in self.keys )

  @property
  def local_keys(self):
    for attr in dir(self):
      if attr.startswith('key_'):
        yield(attr.split('_',1)[1])

  @property
  def keys(self):
    for key in self.local_keys:
      yield key
    for key in self.handlers.keys():
      yield key

  def __getitem__(self, key):
    if key in self.local_keys:
      try:
        return getattr(self, 'key_'+key)()
      except KeyError:
        return None
      except Exception:
        if hydrat.config.getboolean('debug', 'pdb_on_summaryfn_exception'):
          import pdb;pdb.post_mortem()
        else:
          return 'FAILED'
    else: 
      handler = self.handlers[key]
      handler.init(self.result, self.interpreter)
      return handler[key]

  def extend(self, function):
    if hasattr(function, '__class__') and issubclass(function.__class__, Summary):
      # This is an additional summary
      old_keys = set(self.keys)
      new_keys = set(function.keys)
      overlap = old_keys & new_keys
      if len(overlap) != 0:
        raise ValueError, "already have the following keys: " + str(overlap)
      for key in new_keys:
        self.handlers[key] = function
    elif callable(function):
      # This is a wrapped callable
      if function.__name__ in self.keys:
        raise ValueError, "already have the following keys: " + function.__name__
      else:
        setattr(self, 'key_'+ function.__name__, lambda: function(self.result, self.interpreter))
    else:
      raise TypeError, "cannot extend summary with %s" % str(function)


class MicroPRF(Summary):
  def init(self, result, interpreter):
    if result != self.result or interpreter != self.interpreter:
      self.result = result
      self.interpreter = interpreter
      self.micro_score = CombinedMicroAverage(result.overall_confusion_matrix(interpreter))

  def key_micro_precision(self):  return self.micro_score.precision
  def key_micro_recall(self):     return self.micro_score.recall
  def key_micro_fscore(self):     return self.micro_score.fscore

class MacroPRF(Summary):
  def init(self, result, interpreter):
    if result != self.result or interpreter != self.interpreter:
      self.result = result
      self.interpreter = interpreter
      self.macro_score = CombinedMacroAverage(result.overall_confusion_matrix(interpreter))

  def key_macro_precision(self):  return self.macro_score.precision
  def key_macro_recall(self):     return self.macro_score.recall
  def key_macro_fscore(self):     return self.macro_score.fscore

class TimeTaken(Summary):
  def key_avg_learn(self):     return numpy.mean(self.result.individual_metadata('learn_time'))
  def key_avg_classify(self):  return numpy.mean(self.result.individual_metadata('classify_time'))

class Metadata(Summary):
  def __init__(self):
    Summary.__init__(self)
  
  def key_metadata(self): return self.result.metadata

class ClassPRF(Summary):
  def __init__(self, klass):
    self.klass = klass
    Summary.__init__(self)
    self.prf = None

  def init(self, result, interpreter):
    try:
      index = result.class_space.index(self.klass)
      cm = result.overall_confusion_matrix(interpreter).sum(axis=0)[index]
      prf = PRF()(cm)
      self.prf = {'precision':prf[0], 'recall':prf[1], 'fscore':prf[2]}
    except ValueError:
      self.prf = {'precision':None, 'recall':None, 'fscore':None}

  @property
  def keys(self): return iter(['PRF_' + self.klass])

  def __getitem__(self, key):
    if key != 'PRF_' + self.klass:
      raise KeyError
    return self.prf

#####
# Perfold summaries
#####

class PerfoldConfusionMetric(Summary):
  """
  Calculate a confusion matrix metric on a per-fold basis
  """
  def __init__(self):
    Summary.__init__(self)
    self.aggregator = None
    self.metric = None
    self.foldscores = None

  def init(self, result, interpreter):
    fold_results = {}
    for r in result.results:
      index = r.metadata['index']
      cm = r.confusion_matrix(interpreter)
      prf = self.aggregator(cm, self.metric)
      fold_results['fold%d' % index] = {'precision':prf[0], 'recall':prf[1], 'fscore':prf[2]}
    self.foldscores = fold_results

class PerfoldMacroPRF(PerfoldConfusionMetric):
  def __init__(self):
    PerfoldConfusionMetric.__init__(self)
    self.aggregator = Macroaverage()
    self.metric = PRF()

  def key_perfold_macroprf(self):
    return self.foldscores

class PerfoldMicroPRF(PerfoldConfusionMetric):
  def __init__(self):
    PerfoldConfusionMetric.__init__(self)
    self.aggregator = Microaverage()
    self.metric = PRF()

  def key_perfold_microprf(self):
    return self.foldscores

class BoxPlot(PerfoldConfusionMetric):
  def __init__(self, klass, width=300, range=None):
    self.range = range
    self.width = width
    self.klass = klass
    PerfoldConfusionMetric.__init__(self)
    self.aggregator = None
    self.metric = PRF()

  def init(self, result, interpreter):
    index = result.class_space.index(self.klass)
    self.aggregator = lambda x, m: m(x[index])
    PerfoldConfusionMetric.init(self, result, interpreter)

  def key_distplot(self):
    scores = [f['fscore'] for f in self.foldscores.values()]
    return str(markup.oneliner.img(src=boxplot(scores, width=self.width, range=self.range)))
    
def classification_summary():
  sf = Summary()
  sf.extend(MacroPRF())
  sf.extend(MicroPRF())
  sf.extend(Metadata())
  sf.extend(TimeTaken())
  return sf

