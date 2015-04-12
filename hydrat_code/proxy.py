# proxy.py
# Marco Lui February 2011
#

# TODO: Integrate the new SplitArray based FeatureMap and ClassMap back into store,
# and wherever else the old FeatureMap and ClassMap were used.
# TODO: strdoc
# TODO: __str__/__repr__

import logging
import os
import numpy
import scipy.sparse
import multiprocessing as mp

from copy import deepcopy
from itertools import izip, imap

from hydrat import config
from hydrat.common.diskdict import diskdict
from hydrat.common import as_set
from hydrat.store import Store, StoreError, NoData, AlreadyHaveData
from hydrat.inducer import DatasetInducer
from hydrat.datamodel import FeatureMap, ClassMap, TaskSet, DataTask
from hydrat.common.pb import ProgressIter

class DataProxy(TaskSet):
  """
  This class is meant to act as a go-between between the user and the dataset/store
  classes. It is initialized on a dataset (and an optional store), and provides
  convenience methods for accesing portions of that store that are directly
  impacted by the dataset API.
  """
  def __init__( self, dataset, store=None, inducer = None,
        feature_spaces=None, class_space=None, split_name=None, sequence_name=None,
        tokenstream_name=None):
    self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)
    self.dataset = dataset

    if isinstance(store, Store):
      self.store = store
      work_path = os.path.dirname(store.path)
    elif store is None:
      self.store = Store.from_caller()
    else:
      self.store = Store(store, 'a')

    self.inducer = DatasetInducer(self.store) if inducer is None else inducer

    self.feature_spaces = feature_spaces
    self.class_space = class_space
    self.split_name = split_name
    self.sequence_name = sequence_name
    self.tokenstream_name = tokenstream_name

    # Hack to deal with bad interaction between pytables and h5py
    import atexit; atexit.register(lambda: self.store.__del__())

  @property
  def desc(self):
    # TODO: Eliminate this altogether
    return self.metadata

  @property
  def metadata(self):
    md = dict()
    md['dataset'] = self.dsname
    md['instance_space'] = self.instance_space
    md['split'] = self.split_name
    md['sequence'] = self.sequence_name
    md['feature_desc'] = self.feature_desc
    md['class_space'] = self.class_space
    md['feature_combination'] = 'concatenate'
    return md

    
  @property
  def dsname(self):
    return self.dataset.__name__

  def validate_feature_spaces(self, value):
    value = as_set(value)
    if any( not isinstance(s,str) for s in value):
      raise TypeError, "Invalid space identifier: %s" % str(s)
    present_spaces = set(self.dataset.featuremap_names)
    if len(value - present_spaces) > 0:
      # Check if the missing spaces are already in the store
      missing = value - present_spaces
      unknown = []
      for space in missing:
        if not self.store.has_Data(self.dsname, space):
          unknown.append(space)
      if len(unknown) > 0:
        raise ValueError, "Unknown spaces: %s" % str(unknown)
    return value

  @property
  def feature_spaces(self):
    """
    String or sequence of strings representing the feature spaces 
    to operate over.
    """
    return self._feature_spaces

  @feature_spaces.setter
  def feature_spaces(self, value): 
    self._feature_spaces = self.validate_feature_spaces(value)

  @property
  def featurelabels(self):
    """
    List of labels of the feature space
    """
    self.inducer.process(self.dataset, fms=self.feature_spaces)
    labels = []
    # TODO: Handle unlabelled (EG transformed) feature spaces
    for feature_space in sorted(self.feature_spaces):
      labels.extend(self.store.get_Space(feature_space))
    return labels

  @property
  def feature_desc(self):
    return tuple(sorted(self.feature_spaces))

  @property
  def class_space(self):
    """
    String representing the class space to operate over.
    """
    return self._class_space

  @class_space.setter
  def class_space(self, value):
    if value is None:
      self._class_space = None
    else:
      if not isinstance(value, str):
        raise TypeError, "Invalid space identifier: %s" % str(value)
      present_classes = set(self.dataset.classmap_names)
      if value not in present_classes:
        raise ValueError, "Unknown space: %s" % value

      self._class_space = value

  @property
  def classlabels(self):
    self.inducer.process(self.dataset, cms=self.class_space)
    return self.store.get_Space(self.class_space)

  @property
  def split_name(self):
    return self._split_name

  @split_name.setter
  def split_name(self, value):
    if value is None:
      self._split_name=None
    else:
      if not isinstance(value, str):
        raise TypeError, "Invalid split identifier: %s" % str(value)
      present_splits=set(self.dataset.split_names)
      if value not in present_splits:
        raise ValueError, "Unknown split: %s" % value

      self._split_name = value

  @property
  def split(self):
    if self.split_name is None:
      return None
    self.inducer.process(self.dataset, sps=self.split_name)
    return self.store.get_Split(self.dsname, self.split_name)

  @property
  def sequence_name(self):
    return self._sequence_name

  @sequence_name.setter
  def sequence_name(self,value):
    if value is None:
      self._sequence_name=None
    else:
      if not isinstance(value, str):
        raise TypeError, "Invalid sequence identifier: %s" % str(value)
      present_sequences=set(self.dataset.sequence_names)
      if value not in present_sequences:
        raise ValueError, "Unknown sequence: %s" % value

      self._sequence_name = value

  @property
  def sequence(self):
    # TODO: Does this need to be in a SplitArray?
    if self.sequence_name is None:
      return None
    self.inducer.process(self.dataset, sqs=self.sequence_name)
    return self.store.get_Sequence(self.dsname, self.sequence_name)

  @property
  def tokenstream_name(self):
    return self._tokenstream_name

  @tokenstream_name.setter
  def tokenstream_name(self, value):
    if value is None:
      self._tokenstream_name=None
    else:
      if not isinstance(value, str):
        raise TypeError, "Invalid tokenstream identifier: %s" % str(value)
      present_tokenstream=set(self.dataset.tokenstream_names)
      if value not in present_tokenstream and not self.store.has_TokenStreams(self.dsname, value):
        raise ValueError, "Unknown tokenstream: %s" % value

      self._tokenstream_name = value

  @property
  def tokenstream(self):
    self.inducer.process(self.dataset, tss=self.tokenstream_name)
    return self.store.get_TokenStreams(self.dsname, self.tokenstream_name)

  @property
  def instance_space(self):
    # Note that this cannot be set as it is implicit in the dataset
    return self.dataset.instance_space

  @property
  def instancelabels(self):
    self.inducer.process(self.dataset)
    return self.store.get_Space(self.instance_space)

  @property 
  def classmap(self):
    self.inducer.process(self.dataset, cms=self.class_space)
    if self.class_space is None:
      raise ValueError, "class space not set"
    cm = self.store.get_ClassMap(self.dsname, self.class_space)
    return ClassMap(cm.raw, split=self.split, metadata=cm.metadata)
   
  @property
  def featuremap(self):
    self.logger.debug("  loading FM {0}:{1}".format(self.dsname, self.feature_spaces))
    self.inducer.process(self.dataset, fms=self.feature_spaces)

    # TODO: Avoid this duplicate memory consumption
    featuremaps = []
    for feature_space in sorted(self.feature_spaces):
      # TODO: Get rid of this once we introduce new-style featuremaps
      #       into the store
      fm = self.store.get_FeatureMap(self.dsname, feature_space)
      featuremaps.append(FeatureMap(fm.raw, metadata=fm.metadata))

    # Join the featuremaps into a single featuremap
    fm = FeatureMap.union(*featuremaps)
    fm.split = self.split
    return fm

  def process_tokenstream(self, processor):
    """
    Apply a function to each tokenstream and create a new tokenstream
    with the output.
    """
    # Ensure the top-level store has this dataset node.
    # TODO: Refactor this to avoid breaking the abstraction.
    if not hasattr(self.store.datasets, self.dsname):
      self.store.add_Dataset(self.dsname, self.dataset.instance_space, self.dataset.instance_ids)

    if self.tokenstream_name is None:
      raise ValueError, "tokenstream_name not set"

    tokenstream_name = processor.__name__
    if not self.store.has_TokenStreams(self.dsname, tokenstream_name):
      # Read the basis tokenstream
      basis_ts = self.tokenstream
      ids = self.instancelabels
      ts = diskdict(config.getpath('paths','scratch'))

      # Build the new tokenstream
      for i, bts in enumerate(basis_ts):
        ts[ids[i]] = processor(bts)
      
      # Save the new tokenstream
      self.inducer.add_TokenStreams(self.dsname, self.instance_space, tokenstream_name, ts)

    # Set the new tokenstream name
    self.tokenstream_name = tokenstream_name

  def tokenize(self, extractor):
    """
    Map a feature extractor onto a tokenstream and save the corresponding
    output into the backing store.
    """
    # Ensure the top-level store has this dataset node.
    # TODO: Refactor this to avoid breaking the abstraction.
    if not hasattr(self.store.datasets, self.dsname):
      self.store.add_Dataset(self.dsname, self.instance_space, self.dataset.instance_ids)
      
    if self.tokenstream_name is None:
      raise ValueError, "tokenstream_name not set"
    # Definition of space name.
    space_name = '_'.join((self.tokenstream_name,extractor.__name__))
    if not self.store.has_Data(self.dsname, space_name):
      # Read the tokenstream
      tss = self.tokenstream
      feat_dict = diskdict(config.getpath('paths','scratch'))

      # TODO: Backoff behaviour if multiprocessing fails
      #for i, id in enumerate(self.instancelabels):
      #  feat_dict[id] = extractor(tss[i])
      def tokenstream():
        # This hack is to avoid a bad interaction between multiprocessing, progressbar and signals.
        for t in ProgressIter(tss, label=extractor.__name__):
          yield t

      if config.getboolean('parameters','parallel_tokenize'):
        pool = mp.Pool(config.getint('parameters','job_count'))
        tokens = pool.imap(extractor, tokenstream())
      else:
        tokens = imap(extractor, tokenstream())

      #for i, id in enumerate(self.instancelabels):
      for id, inst_tokens in izip(self.instancelabels, tokens):
        feat_dict[id] = dict(inst_tokens)

        if len(inst_tokens) == 0:
          msg =  "%s(%s) has no tokens for '%s'" % (self.tokenstream, extractor.__name__, id)
          if config.getboolean('debug','allow_empty_instance'):
            self.logger.warning(msg)
          else:
            raise ValueError, msg

      self.inducer.add_Featuremap(self.dsname, self.instance_space, space_name, feat_dict)

    self.feature_spaces = space_name

  def __len__(self):
    if self.split_name is None:
      return 0
    else:
      return self.split.shape[1]

  def __getitem__(self, key):
    fm = self.featuremap
    cm = self.classmap
    sq = self.sequence

    fold = fm.folds[key]
    t = DataTask(fm.raw, cm.raw, fold.train_ids, fold.test_ids, 
        {'index':key}, sequence=sq)
    return t

  @property
  def taskset(self):
    return self.store.new_TaskSet(self)

class CrossDomainDataProxy(DataProxy):
  def __init__(self, train_ds, eval_ds, store=None, feature_spaces=None,
        class_space=None, sequence_name=None, tokenstream_name=None):
    """
    Initialize on two datasets, one for training and one
    for evaluation. Note that we want these two to share
    the same Store. This behaves just like a DataProxy,
    but it bridges two datasets. The feature and class
    spaces stay the same, but the instance space is defined
    as the concatenation of the two.
    """
    self.train = DataProxy(train_ds, store, feature_spaces=feature_spaces,
        class_space=class_space, sequence_name=sequence_name, 
        tokenstream_name=tokenstream_name)
    self.eval = DataProxy(eval_ds, self.train.store, feature_spaces=feature_spaces,
        class_space=class_space, sequence_name=sequence_name, 
        tokenstream_name=tokenstream_name)

    self.feature_spaces = feature_spaces
    self.class_space = class_space
    self.sequence_name = sequence_name
    self.tokenstream_name = tokenstream_name


    self.inducer = self.train.inducer
    self.store = self.train.store

  @property
  def dsname(self):
    return '+'.join((self.train.dsname, self.eval.dsname))

  @property
  def feature_spaces(self):
    return self._feature_spaces

  @feature_spaces.setter
  def feature_spaces(self, value):
    self.train.feature_spaces = value
    self.eval.feature_spaces = value
    self._feature_spaces = as_set(value)

  @property
  def featurelabels(self):
    self.inducer.process(self.train.dataset, fms=self.feature_spaces)
    self.inducer.process(self.eval.dataset, fms=self.feature_spaces)
    labels = []
    for feature_space in sorted(self.feature_spaces):
      labels.extend(self.store.get_Space(feature_space))
    return labels

  @property
  def class_space(self):
    return self._class_space

  @property
  def classlabels(self):
    self.inducer.process(self.train.dataset, cms=self.class_space)
    self.inducer.process(self.eval.dataset, cms=self.class_space)
    return self.store.get_Space(self.class_space)

  @class_space.setter
  def class_space(self, value):
    self.train.class_space = value
    try:
      self.eval.class_space = value
    except ValueError:
      # Eval dataset may not know about the class space, for example in
      # a shared task where the goldstandard for test data is not given.
      pass
    self._class_space = value

  @property
  def split_name(self):
    return 'crossdomain'

  @property
  def split(self):
    num_train = len(self.train.instancelabels)
    num_eval = len(self.eval.instancelabels)
    num_inst = num_train + num_eval
    retval = numpy.zeros((num_inst,1,2), dtype=bool)
    retval[:num_train,:,0] = True
    retval[-num_eval:,:,1] = True
    return retval

  @property
  def tokenstream_name(self):
    return self._tokenstream_name

  @tokenstream_name.setter
  def tokenstream_name(self, value):
    if value is None:
      self._tokenstream_name=None
    else:
      if not isinstance(value, str):
        raise TypeError, "Invalid tokenstream identifier: %s" % str(value)
      present_tokenstream_train=set(self.train.dataset.tokenstream_names)
      present_tokenstream_eval=set(self.eval.dataset.tokenstream_names)
      if value not in present_tokenstream_train:
        raise ValueError, "Unknown tokenstream: %s" % value
      if value not in present_tokenstream_eval:
        raise ValueError, "Unknown tokenstream: %s" % value

      self._tokenstream_name = value

  @property
  def tokenstream(self):
    return self.train.tokenstream + self.eval.tokenstream

  @property
  def instance_space(self):
    """ We only return the eval space, since it is what the results will be over."""
    return self.eval.instance_space

  @property
  def instancelabels(self):
    return self.train.instancelabels + self.eval.instancelabels

  @property
  def classmap(self):
    cm_train = self.train.classmap
    try:
      cm_eval_raw = self.eval.classmap.raw
    except ValueError:
      # Create a stub blank classmap
      # TODO: Do we need to somehow note this in the metadata that the
      #       eval space had no data for the class?
      eval_doc_count = len(self.eval.instancelabels)
      class_count = cm_train.raw.shape[1]
      cm_eval_raw = numpy.zeros((eval_doc_count,class_count), dtype=bool)

    raw = numpy.vstack((cm_train.raw, cm_eval_raw))
    md = dict(dataset=self.dsname, class_space=self.class_space, 
        instance_space=self.instance_space)
    return ClassMap(raw, split=self.split, metadata=md)

  @property
  def featuremap(self):
    feats_train = self.train.featurelabels
    feats_eval = self.eval.featurelabels

    if len(set(feats_train) - set(feats_eval)) != 0:
      raise ValueError("train contains features that eval does not!")

    fm_train=self.train.featuremap.raw
    fm_eval=self.eval.featuremap.raw

    size_diff = fm_eval.shape[1] - fm_train.shape[1]
    if size_diff > 0:
      # We need to upsize the train map to allow it to stack.
      fm_train = scipy.sparse.csr_matrix((fm_train.data, fm_train.indices, fm_train.indptr), shape=(fm_train.shape[0],fm_eval.shape[1]))

    raw = scipy.sparse.vstack((fm_train, fm_eval)).tocsr()
    md = dict(dataset=self.dsname, feature_spaces=self.feature_spaces, 
        instance_space=self.instance_space)
    return FeatureMap(raw, split=self.split, metadata=md)

  def tokenize(self, extractor):
    # TODO: How does this broadcast?
    raise NotImplementedError


class TransductiveLOO(DataProxy):
  """
  Transductive transfer learning. 
  We do this as leave-one-out over the set of domains.
  This is similar to DomainCrossValidation, but is initialized with
  a list of proxies rather than a list of datasets.
  Note that this was previously incorrectly called InductiveLOO.
  The name has been changed to reflect standard terminology.
  """

  def __init__(self, proxies):
    for proxy in proxies:
      if not isinstance(proxy, DataProxy):
        raise TypeError, "all proxies must be DataProxy instances"
      if not proxy.feature_desc == proxies[0].feature_desc:
        raise ValueError, "all proxies must share a single feature desc"
      if not proxy.class_space == proxies[0].class_space:
        raise ValueError, "all proxies must share a single class space"
      if proxy.sequence is not None:
        raise ValueError, "no support for dealing with sequencing"
      if proxy.split is not None:
        raise ValueError, "no support for dealing with splits"
      self.proxies  = proxies
    self._fm = None
    self._cm = None

  @property
  def metadata(self):
    md = dict()
    md['dataset'] = self.dsname
    md['instance_space'] = self.instance_space
    md['split'] = self.split_name
    md['sequence'] = self.sequence_name
    md['feature_desc'] = self.feature_desc
    md['class_space'] = self.class_space
    return md

  @property
  def dsname(self):
    return "TransductiveLOO"

  @property
  def sequence(self):
    return None

  @property
  def sequence_name(self):
    return None

  @property
  def featurelabels(self):
    return self.proxies[0].featurelabels

  @property
  def feature_desc(self):
    return self.proxies[0].feature_desc

  @property
  def class_space(self):
    return self.proxies[0].class_space

  @property
  def classlabels(self):
    return self.proxies[0].classlabels
  
  @property
  def instance_space(self):
    return tuple(p.instance_space for p in self.proxies) 

  @property
  def instancelabels(self):
    retval = []
    for p in self.proxies:
      dsname = p.dsname
      retval.extend('{0}-{1}'.format(dsname,l) for l in p.instancelabels)
    assert len(set(retval)) == len(retval), "duplicate instancelabels"
    return retval

  @property
  def split_name(self):
    return "TransductiveLOO"

  @property
  def split(self):
    # we need this to implement featuremap and classmap
    num_inst = len(self.instancelabels)
    num_fold = len(self.proxies)
    
    # We need to mark the role of each document in each fold
    test_inst = numpy.zeros((num_inst, num_fold), dtype=bool)

    # For each proxy in turn, we mark all of its documents as the
    # test documents
    start = 0
    for i,p in enumerate(self.proxies):
      end = start + len(p.instancelabels)
      test_inst[numpy.arange(start, end), i, :] = True
      start = end

    # We then mark all the non-test documents as train documents for
    # each fold
    train_inst = numpy.logical_not(test_inst)

    retval = numpy.dstack((train_inst, test_inst))
    return retval

  @property
  def domainmap(self):
    """
    ClassMap over domains.
    """
    # we use the test-domain membership map in the split as the cm itself
    retval = ClassMap(self.split[...,1], self.split)
    return retval

  @property 
  def classmap(self):
    if self._cm is None:
      cm = retval = ClassMap.stack(*(p.classmap for p in self.proxies))
      cm.split = self.split
      self._cm = cm
    return self._cm

  @property
  def featuremap(self):
    if self._fm is None:
      fm = FeatureMap.stack(*(p.featuremap for p in self.proxies))
      fm.split = self.split
      self._fm = fm
    return self._fm

  def __getitem__(self, key):
    fm = self.featuremap
    cm = self.classmap
    sq = self.sequence

    domains = [ p.dsname for p in self.proxies ]
    fold = fm.folds[key]
    metadata = dict()
    metadata['index'] = key
    metadata['domain.train'] = tuple(d for j,d in enumerate(domains) if key != j)
    metadata['domain.test'] = domains[key],
    t = DataTask(fm.raw, cm.raw, fold.train_ids, fold.test_ids, 
        metadata, sequence=sq)
    return t


    


class DomainCrossValidation(DataProxy):
  """
  Cross-validate across a set of domains.
  """
  def __init__(self, datasets, store=None, feature_spaces=None,
        class_space=None, sequence_name=None, tokenstream_name=None):
    ds, datasets = datasets[0], datasets[1:]
    proxy = DataProxy(ds, store, feature_spaces=feature_spaces,
        class_space=class_space, sequence_name=sequence_name, 
        tokenstream_name=tokenstream_name)
    self.proxies = [ proxy ]

    self.inducer = self.proxies[0].inducer
    self.store = self.proxies[0].store
    
    # Build up a list of proxies, one per dataset
    # They can all share the store and inducer
    for ds in datasets:
      proxy = DataProxy(ds, self.store, self.inducer,
          feature_spaces=feature_spaces, class_space=class_space, 
          sequence_name=sequence_name, tokenstream_name=tokenstream_name)
      self.proxies.append(proxy)
    # Sort proxies by their dataset name to avoid identifying different
    # orderings as different tasksets
    self.proxies.sort(key=lambda x:x.dsname)

    self.feature_spaces = feature_spaces
    self.class_space = class_space
    self.sequence_name = sequence_name
    self.tokenstream_name = tokenstream_name

  @property
  def dsname(self):
    return '+'.join(p.dsname for p in self.proxies)

  @property
  def feature_spaces(self):
    return self._feature_spaces

  @feature_spaces.setter
  def feature_spaces(self, value):
    for p in self.proxies:
      p.feature_spaces = value
    self._feature_spaces = as_set(value)

  @property
  def featurelabels(self):
    for p in self.proxies:
      self.inducer.process(p.dataset, fms=self.feature_spaces)
    labels = []
    for feature_space in sorted(self.feature_spaces):
      labels.extend(self.store.get_Space(feature_space))
    return labels

  @property
  def class_space(self):
    return self._class_space

  @property
  def classlabels(self):
    for p in self.proxies:
      self.inducer.process(p.dataset, cms=self.class_space)
    return self.store.get_Space(self.class_space)

  @class_space.setter
  def class_space(self, value):
    for p in self.proxies:
      p.class_space = value
    self._class_space = value

  @property
  def split_name(self):
    return 'DomainCrossValidation'

  @property
  def split(self):
    """
    Leave-one-out cross-validation of domains
    """
    num_domains = len(self.proxies)
    num_inst = sum(len(p.instancelabels) for p in self.proxies)

    start_index = [ 0 ]
    for p in self.proxies:
      start_index.append( start_index[-1] + len(p.instancelabels) )

    retval = numpy.zeros((num_inst,num_domains,2), dtype=bool)
    for i in xrange(num_domains):
      retval[start_index[i]:start_index[i+1],i,1] = True # Set Eval
      retval[:,i,0] = numpy.logical_not(retval[:,i,1]) #Train on all that are not eval
    return retval

  @property
  def tokenstream_name(self):
    return self._tokenstream_name

  @tokenstream_name.setter
  def tokenstream_name(self, value):
    for p in self.proxies:
      p.tokenstream_name = value
    self._tokenstream_name = value

  @property
  def tokenstream(self):
    ts = []
    for p in self.proxies:
      ts.extend(p.tokenstream)
    return ts

  @property
  def instance_space(self):
    """ Returns a concatenation of the two instance spaces """
    return '+'.join(p.instance_space for p in self.proxies)

  @property
  def instancelabels(self):
    # TODO: May need to handle clashes in labels. Could prefix dataset name.
    labels = []
    for p in self.proxies:
      labels.extend(p.instancelabels)
    return labels

  @property
  def classmap(self):
    cms = [ p.classmap.raw for p in self.proxies ]
    raw = numpy.vstack(cms)
    md = dict(dataset=self.dsname, class_space=self.class_space, 
        instance_space=self.instance_space)
    return ClassMap(raw, split=self.split, metadata=md)

  @property
  def featuremap(self):
    # NOTE: We access the featurelabels in order to ensure that
    # full common feature space is learned before we attempt to access
    # the actual featuremaps
    for p in self.proxies:
      p.featurelabels

    fms = [ p.featuremap.raw for p in self.proxies ]
    raw = scipy.sparse.vstack(fms).tocsr()
    md = dict(dataset=self.dsname, feature_spaces=self.feature_spaces, 
        instance_space=self.instance_space)
    return FeatureMap(raw, split=self.split, metadata=md)

  def tokenize(self, extractor):
    # TODO: How does this broadcast?
    raise NotImplementedError

from hydrat.experiment import Experiment
from hydrat.classifier.meta.featurestacking import StackedResult

class StackingProxy(DataProxy):
  """
  A hydrat proxy subclass that implements an internal stacking
  metalearner.

  Internally, this maintains a StackedResult instance to serve 
  Tasks from. This StackedResult will be generated on-demand, and
  will be cached unless invalidated by a change in parameters.
  """
  def __init__( self, dataset, learner, split_name='crossvalidation', **kwargs):
    #if 'split_name' not in kwargs:
    #  raise ValueError("split name must be provided")
    DataProxy.__init__(self, dataset, **kwargs)
    self.learner = learner
    self.__taskset = None # The internal StackedResult

  def init_taskset(self):
    # TODO: Add debug output!
    if self.__taskset is None:
      store = self.store

      # Obtain the necessary results
      proxy = DataProxy(self.dataset, store=store)
      proxy.class_space = self.class_space
      proxy.split_name = 'crossvalidation'

      tsrs = []
      for fs in self.feature_spaces:
        proxy.feature_spaces = fs
        e = Experiment(proxy, self.learner)
        tsrs.append(store.new_TaskSetResult(e))

      # Compile a suitable taskset

      # TODO: Refactor against hydrat.browser.result:244
      # Compute the set of keys present in the metadata over all results 
      all_keys = sorted(reduce(set.union, (set(t.metadata.keys()) for t in tsrs)))
      # Compute the set of possible values for each key 
      values_set = {}
      for k in all_keys:
        for t in tsrs:
          try:
            values_set[k] = set(t.metadata.get(k,'UNKNOWN') for t in tsrs)
          except TypeError:
            # skip unhashable
            pass
      # Compute the set of key-values which all the results have in common
      md = dict( (k, values_set[k].pop()) for k in values_set if len(values_set[k]) == 1)
      #md['feature_desc'] = tuple(sorted(sum((d['feature_desc'] for d in descs), tuple())))
      #md['stacking_desc'] = descs
      md['feature_desc'] = self.feature_desc
      md['feature_combination'] = 'stacking'

      self.__taskset = StackedResult(tsrs, md)

  @property
  def metadata(self):
    # TODO: should be able to compile metadata without needing to assemble the inner taskset
    self.init_taskset()
    return self.__taskset.metadata

  @property
  def feature_spaces(self):
    """
    String or sequence of strings representing the feature spaces 
    to operate over.
    """
    return self._feature_spaces

  @feature_spaces.setter
  def feature_spaces(self, value): 
    self._feature_spaces = DataProxy.validate_feature_spaces(self, value)
    self.__taskset = None #invalidate existing taskset

  def __getitem__(self, key):
    self.init_taskset()
    return self.__taskset[key]

  def __len__(self):
    self.init_taskset()
    return len(self.__taskset)

  def __iter__(self):
    for i in xrange(len(self)):
      yield self[i]

  def __contains__(self, key):
    return 0 <= key < len(self)

