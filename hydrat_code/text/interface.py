import numpy

from hydrat.datamodel import TaskSetResult, Result
from hydrat.common.pb import ProgressBar, get_widget
from hydrat.common.mapmatrix import map2matrix
from hydrat.common.timer import Timer

class TextClassifier(object):
  """
  Base class for pre-trained external classifiers
  that accept raw text and produce a prediction.
  """
  def __init__(self, label_map=None):
    """
    @param label_map a function that maps the output of the classifier 
                     onto the final label
    """
    self.label_map = label_map if label_map is not None else lambda x: x

  def classify(self, text):
    raise NotImplementedError("deriving class must implement this")

  def classify_batch(self, texts, callback=None):
    retval = []
    for i, t in enumerate(texts):
      retval.append(self.label_map(self.classify(t)))
      if callback is not None:
        callback(i)
    return retval

  def __call__(self, text):
    return self.label_map(self.classify(text))

class ProxyExperiment(TaskSetResult):
  """
  Experiment implementation using a TextClassifier. The data to be
  classified is accessed via a DataProxy instance.
  """
  def __init__(self, classifier, proxy):
    self.classifier = classifier
    self.proxy = proxy

  @property
  def metadata(self):
    keys = ['class_space','dataset','instance_space','learner','learner_params']
    md = dict()
    for key in keys:
      md[key] = self.classifier.metadata.get(key, None)
    md['eval_dataset'] = self.proxy.dsname
    md['eval_space']   = self.proxy.instance_space
    return md

  @property
  def results(self):
    proxy = self.proxy
    with ProgressBar(widgets=get_widget(self.classifier.__class__.__name__),maxval=len(self.proxy.instancelabels)) as pb:
      with Timer() as t:
        klass = self.classifier.classify_batch(self.proxy.tokenstream, callback=pb.update)
    class_map = dict(zip(proxy.instancelabels, klass))

    cl = map2matrix(class_map, proxy.instancelabels, proxy.classlabels)
    gs = proxy.classmap.raw
    instance_indices = numpy.arange(len(proxy.instancelabels))
    md = dict(learn_time=None, classify_time=t.elapsed)
    return [ Result(gs, cl, instance_indices, md ) ]

class DatasetExperiment(TaskSetResult):
  """
  Experiment implementation using a TextClassifier. The data to be
  classified is accessed via a Dataset instance. This avoids the 
  overhead of caching the dataset into the .h5 file that is normally
  incurred with the DataProxy interface.
  """
  def __init__(self, classifier, dataset, classmap=None, tokenstream='byte'):
    if classmap is None:
      raise ValueError("classmap must be specified")
    if not classmap in set(dataset.classspace_names):
      raise ValueError("no classmap {0} in dataset".format(classmap))
    if not tokenstream in set(dataset.tokenstream_names):
      raise ValueError("no tokenstream {0} in dataset".format(tokenstream))
    self.classifier = classifier
    self.ds = dataset
    self.cm = classmap
    self.ts = tokenstream

  @property
  def metadata(self):
    keys = ['class_space','dataset','instance_space','learner','learner_params']
    md = dict()
    for key in keys:
      md[key] = self.classifier.metadata.get(key, None)
    md['eval_dataset'] = self.ds.__name__
    return md

  @property
  def results(self):
    ids = self.ds.instance_ids
    ts = self.ds.tokenstream(self.ts)
    classlabels = self.ds.classspace(self.cm)

    # Do the actual classification
    texts_iter = ( ts[i] for i in ids )
    with ProgressBar(widgets=get_widget(self.classifier.__class__.__name__),maxval=len(ids)) as pb:
      with Timer() as t:
        klass = self.classifier.classify_batch(texts_iter, callback=pb.update)
    class_map = dict(zip(ids, klass))

    cl = map2matrix(class_map, ids, classlabels)
    gs = map2matrix(self.ds.classmap(self.cm), ids, classlabels)
    instance_indices = numpy.arange(len(ids))
    md = dict(learn_time=None, classify_time=t.elapsed)
    return [ Result(gs, cl, instance_indices, md ) ]
