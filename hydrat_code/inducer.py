import logging
import numpy
from hydrat import config
from hydrat.store import NoData, AlreadyHaveData
from hydrat.common.mapmatrix import map2matrix
from hydrat.common.pb import ProgressIter
from hydrat.common import as_set
from hydrat.common.sequence import sequence2matrix
from hydrat.task.sampler import membership_vector
from hydrat.common.disklist import disklist

logger = logging.getLogger(__name__)


from collections import defaultdict
class Enumerator(object):
  """
  Enumerator object. Returns a larger number each call. 
  Can be used with defaultdict to enumerate a sequence of items.
  """
  def __init__(self, start=0):
    self.n = start

  def __call__(self):
    retval = self.n
    self.n += 1
    return retval

class DatasetInducer(object):

  def __init__(self, store):
    self.store = store

  def process(self, dataset, fms=None, cms=None, tss=None, sqs=None, sps=None):
    dsname = dataset.__name__
    instance_space = dataset.instance_space

    # Work out if this is the first time we encounter this dataset
    if self.store.has_Dataset(dsname):
      logger.debug("{0}: already had dataset".format(dsname))
    else:
      logger.debug("{0}: adding new dataset".format(dsname))
      self.store.add_Dataset(dsname, dataset.instance_space, dataset.instance_ids)

    fms = as_set(fms)
    cms = as_set(cms)
    tss = as_set(tss)
    sqs = as_set(sqs)
    sps = as_set(sps)

    present_fm = set(self.store.list_FeatureSpaces(dsname))
    present_cm = set(self.store.list_ClassSpaces(dsname))
    present_ts = set(self.store.list_TokenStreams(dsname))
    present_sq = set(self.store.list_Sequence(dsname))
    present_sp = set(self.store.list_Split(dsname))

    logger.debug("  present_fm: ({0}) {1}".format(len(present_fm),str(present_fm)))
    logger.debug("  present_cm: ({0}) {1}".format(len(present_cm),str(present_cm)))
    logger.debug("  present_ts: ({0}) {1}".format(len(present_ts),str(present_ts)))
    logger.debug("  present_sq: ({0}) {1}".format(len(present_sq),str(present_sq)))
    logger.debug("  present_sp: ({0}) {1}".format(len(present_sp),str(present_sp)))

    # Handle explicit class spaces 
    for key in set(dataset.classspace_names):
      logger.debug("  processing explicit class space '%s'", key)
      if self.store.has_Space(key):
        logger.debug('  - already have space %s', key)
      else:
        c_metadata = {'type':'class','name':key}
        self.store.add_Space(dataset.classspace(key), c_metadata)

    # Handle all the class maps
    for key in cms - present_cm:
      logger.debug("  processing class map '%s'", key)
      try:
        self.add_Classmap(dsname, instance_space, key, dataset.classmap(key))
      except AlreadyHaveData,e :
        logger.debug(e)

    # Handle all the feature maps
    for key in fms - present_fm:
      logger.debug("  processing feature map '%s'", key)
      try:
        self.add_Featuremap(dsname, instance_space, key, dataset.featuremap(key, cache=False) )
      except AlreadyHaveData,e :
        logger.warning(e)
        # TODO: Why are we calling pdb for this?
        import pdb;pdb.post_mortem()
      except AttributeError,e :
        logger.warning(e)
        if config.getboolean('debug','pdb_on_invalid_fm'):
          import pdb;pdb.post_mortem()
        else:
          pass

    # Handle all the token streams
    for key in tss - present_ts:
      logger.debug("  processing token stream '%s'", key)

      try:
        self.add_TokenStreams(dsname, instance_space, key, dataset.tokenstream(key))
      except AlreadyHaveData,e :
        # TODO: I don't think this is actually raised by anything.
        logger.warning(e)

    # Handle all the sequences
    for key in sqs - present_sq:
      logger.debug("  processing sequence '%s'", key)
      self.add_Sequence(dsname, instance_space, key, dataset.sequence(key))

    # Handle all the splits
    for key in sps - present_sp:
      logger.debug("  processing split '%s'", key)
      self.add_Split(dsname, instance_space, key, dataset.split(key))


  def add_Split(self, dsname, instance_space, split_name, split):
    instance_ids = self.store.get_Space(instance_space)
    if 'train' in split and 'test' in split:
      # Train/test type split.
      train_ids = membership_vector(instance_ids, split['train'])
      test_ids = membership_vector(instance_ids, split['test'])
      spmatrix = numpy.dstack((train_ids, test_ids)).swapaxes(0,1)

    elif all(key.startswith('fold') for key in split):
      # Cross-validation folds
      folds_present = sorted(split)
      partitions = []
      for fold in folds_present:
        test_ids = membership_vector(instance_ids, split[fold])
        train_docids = sum((split[f] for f in folds_present if f is not fold), [])
        train_ids = membership_vector(instance_ids, train_docids)
        partitions.append( numpy.dstack((train_ids, test_ids)).swapaxes(0,1) )
      spmatrix = numpy.hstack(partitions)

    elif all(key.startswith('learncurve') for key in split):
      # Learning curve. learncurve0 ... learncurve(N) are the training
      # portions, where (N) is a number. There will be N+1 tasks, where 
      # task m has training data learncurve0 ... learncurve(m). The test
      # data is marked with a special key, learncurveT, using just the
      # capital letter T.
      if 'learncurveT' not in split:
        raise ValueError("missing test data for learning curves")
      test_ids = membership_vector(instance_ids, split['learncurveT'])
      partitions = []
      for i in range(1,len(split)):
        train_docids = sum((split["learncurve{0}".format(j)] for j in range(i)), [])
        train_ids = membership_vector(instance_ids, train_docids)
        partitions.append( numpy.dstack((train_ids, test_ids)).swapaxes(0,1) )
      spmatrix = numpy.hstack(partitions)

    else:
      raise ValueError, "Unknown type of split" 

    self.store.add_Split(dsname, split_name, spmatrix)


  def add_Sequence(self, dsname, instance_space, seq_name, sequence):
    # This involves converting the sequence representation from lists of identifers 
    # in-dataset identifiers to a matrix. 
    # Axis 0 represents the parent and axis 1 represents the child.
    # A True value indicates a directed edge from parent to child.
    instance_ids = self.store.get_Space(instance_space)
    index = dict((k,i) for i,k in enumerate(instance_ids))
    sqlist = [ [index[id] for id in s] for s in sequence ]
    sqmatrix = sequence2matrix(sqlist) 
    logger.debug("Adding Sequence'%s' to Dataset '%s'", seq_name, dsname)
    self.store.add_Sequence(dsname, seq_name, sqmatrix)

  def add_TokenStreams(self, dsname, instance_space, stream_name, tokenstreams):
    metadata = dict()
    instance_ids = self.store.get_Space(instance_space)

    tslist = [tokenstreams[i] for i in instance_ids]
    logger.debug("Adding Token Stream '%s' to Dataset '%s'", stream_name, dsname)
    self.store.add_TokenStreams(dsname, stream_name, tslist)

  def add_Featuremap(self, dsname, instance_space, space_name, feat_dict):
    # Check the list of instances is correct
    instance_ids = self.store.get_Space(instance_space)
    assert set(instance_ids) == set(feat_dict.keys())

    # load up the existing space if any
    try:
      space = self.store.get_Space(space_name)
      if len(space) != len(set(space)):
        raise ValueError, "space %s was not unique!" % space_name
      logger.debug('existing space "%s" (%d feats)', space_name, len(space))
    except NoData:
      space = []
      logger.debug('new space "%s', space_name)
      
    # Enumerate the existing keys
    feat_index = defaultdict(Enumerator())
    [ feat_index[f] for f in space ]

    assert (len(feat_index) == len(space))

    logger.debug("Computing feature map")
    # Build a list of triplets:
    # (instance#, feat#, value)
    feat_map = disklist(config.getpath('paths','scratch'))

    max_feats = config.getint('parameters','max_feats')
    feat_sum = defaultdict(int)

    for i, id in enumerate(ProgressIter(instance_ids,label='FeatureMap(%s)' % space_name)):
      d = feat_dict[id]
      for feat in d:
        j = feat_index[feat]
        feat_sum[j] += d[feat]
        feat_map.append((i,j,d[feat]))
    logger.debug(' space "%s" now has %d unique features', space_name, len(feat_index))

    if len(feat_index) > max_feats:
      logger.debug("culling features, keeping top %d features", max_feats)
      selected = set(sorted(feat_sum, key=feat_sum.get, reverse=True)[:max_feats])
      fm = feat_map
      feat_map = disklist(config.getpath('paths','scratch'))
      for t in fm:
        if t[1] in selected:
          feat_map.append(t)
      culled_count = len(fm) - len(feat_map)
      prop = float(culled_count) / len(fm)
      logger.debug("eliminated %d triplets (%.1f%%)", culled_count, prop)


    # Store the extended space
    space = sorted(feat_index, key=feat_index.get)
    if self.store.has_Space(space_name):
      logger.debug('  extending exisitng space "%s"', space_name)
      self.store.extend_Space(space_name, space)
    else:
      logger.debug('  saving new space "%s"', space_name)
      self.store.add_Space(space, {'type':'feature','name':space_name})

    logger.debug("adding map to store")
    self.store.add_FeatureDict(dsname, space_name, feat_map)

  def add_Classmap(self, dsname, instance_space, space_name, docclassmap):
    instance_ids = self.store.get_Space(instance_space)
    if not config.getboolean('debug','allow_str_classset'):
      if any(isinstance(d, str) or isinstance(d, unicode) for d in docclassmap.values()):
        raise ValueError, "str detected as classset - did you forget to wrap classmap values in a list?"
      
    classlabels = reduce(set.union, (set(d) for d in docclassmap.values()))
    c_metadata = {'type':'class','name':space_name}
    try:
      c_space = self.store.get_Space(space_name)
      #TODO: Being a subset of a stored space is OK!!! - has this been sorted?
      if not classlabels <= set(c_space):
        raise ValueError, "Superfluous classes: %s" % (classlabels - set(c_space))
      # Replace with the stored one, as ordering is important
      classlabels = c_space
    except NoData:
      classlabels = sorted(classlabels)
      self.store.add_Space(classlabels, c_metadata)

    if self.store.has_Data(dsname, space_name):
      raise ValueError, "Already have data for dataset '%s' in space '%s'"% (dsname, space_name)

    class_map = map2matrix(docclassmap, instance_ids, classlabels)
    self.store.add_ClassMap(dsname, space_name, class_map)

