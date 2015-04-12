import uuid
import time
import tables
import os
import warnings
import logging
import UserDict 
import numpy
warnings.simplefilter("ignore", tables.NaturalNameWarning)
import itertools
from scipy.sparse import lil_matrix, csr_matrix

from hydrat import config
from hydrat.common import progress
from hydrat.common.metadata import metadata_matches, get_metadata
from hydrat.datamodel import FeatureMap, ClassMap, Result, TaskSetResult
from hydrat.datamodel import Task, TaskSet
from hydrat.datamodel import Result, TaskSetResult
from hydrat.common.pb import ProgressIter

from hydrat.common.decorators import deprecated
from hydrat.common.filelock import FileLock, FileLockException
from hydrat.common.timer import Timer

logger = logging.getLogger(__name__)

class StoreError(Exception): pass
class NoData(StoreError): pass
class AlreadyHaveData(StoreError): pass
class InsufficientMetadata(StoreError): pass

# TODO: Provide a datamodel abstraction for datasets 

# Features are internally stored as sparse arrays, which are serialized at the
# pytables level to tables of instance, feature, value triplets. We support
# both Integer and Real features.

class IntFeature(tables.IsDescription):
  ax0    = tables.UInt64Col()
  ax1    = tables.UInt64Col()
  value  = tables.UInt64Col()

class RealFeature(tables.IsDescription):
  ax0    = tables.UInt64Col()
  ax1    = tables.UInt64Col()
  value  = tables.Float64Col()

class BoolFeature(tables.IsDescription):
  ax0    = tables.UInt64Col()
  ax1    = tables.UInt64Col()
  value  = tables.BoolCol()
# TODO: Declare a configurable compression filter
#         tables.Filters(complevel=5, complib='zlib') 

STORE_VERSION = 4

def update_h5store(fileh):
  """
  Update the format of a h5file-backed store from an earlier version.
  This is done as an incremental procress, i.e. an update 0->2 is done as
  0->1->2
  The fileh is assumed to be writeable.
  An update from a file without version number is taken to mean that is is either
  version unknown or a brand new file.
  """
  logger.debug("Running update_h5store")
  root = fileh.root

  version = root._v_attrs['version'] if 'version' in root._v_attrs else 0
  
  if version < 1:
    # No version, or new file
    # Ensure that the major nodes exist
    logger.debug('updating to version 1')
    for node in ['spaces', 'datasets', 'tasksets', 'results']:
      if not hasattr(root, node):
        fileh.createGroup( root, node )
    # Check that the dataset nodes are well-formed
    for dsnode in root.datasets:
      if not hasattr(dsnode, 'tokenstreams'):
        logger.debug('Node %s did not have tokenstreams node; adding.', dsnode._v_name)
        fileh.createGroup( dsnode, "tokenstreams" )
      if not hasattr(dsnode, 'sequence'):
        logger.debug('Node %s did not have sequence node; adding.', dsnode._v_name)
        fileh.createGroup( dsnode, "sequence" )
  if version < 2:
    # In version 2, we introduce the concept of instance spaces, detaching the instance
    # identifiers from the dataset nodes and instead attaching them to the space nodes
    logger.debug('updating to version 2')
    for dsnode in root.datasets:
      # Move the instance id node to spaces
      id_node = dsnode.instance_id
      id_node._v_attrs['size']      = len(dsnode.instance_id)
      id_node._v_attrs['type']      = 'instance'
      id_node._v_attrs['name']      = dsnode._v_name
      id_node._v_attrs['encoding']  = 'utf8' # to be safe, in case we had e.g. utf8 filenames
      fileh.moveNode(dsnode.instance_id, root.spaces, dsnode._v_name)
      # Unless otherwise specified, the instance space is the dataset name
      dsnode._v_attrs['instance_space'] = dsnode._v_name

    # Add the instance space metadata to all tasksets
    for tsnode in root.tasksets:
      tsnode._v_attrs.instance_space = tsnode._v_attrs.dataset
      for t in tsnode:
        t._v_attrs.instance_space = t._v_attrs.dataset
        
    # Add the instance space metadata to all results
    for rnode in root.results:
      rnode._v_attrs.instance_space = rnode._v_attrs.dataset
      if hasattr(rnode._v_attrs, 'eval_dataset'):
        rnode._v_attrs.eval_space = rnode._v_attrs.eval_dataset
      for node in rnode:
        if node._v_name == 'summary':
          for summary in node:
            summary._v_attrs.instance_space = summary._v_attrs.dataset
            if hasattr(summary._v_attrs, 'eval_dataset'):
              summary._v_attrs.eval_space = summary._v_attrs.eval_dataset
        else:
          node._v_attrs.instance_space = node._v_attrs.dataset
          if hasattr(node._v_attrs, 'eval_dataset'):
            node._v_attrs.eval_space = node._v_attrs.eval_dataset
  if version < 3:
    # In version 3, we add weights associated with task nodes
    for tsnode in root.tasksets:
      for t in tsnode:
        fileh.createGroup(t, 'weights')
  if version < 4:
    # In version 4, we introduced a node to store splits in datasets
    for dsnode in root.datasets:
      if not hasattr(dsnode, 'split'):
        logger.debug('Node %s did not have split node; adding.', dsnode._v_name)
        fileh.createGroup( dsnode, "split" )
  # TODO:
  # Replace all boolean maps for tasks with their equivalent flatnonzero indices
  # Eliminate UUID from taskset and result metadata
  # Get rid of all date attrs
  # Ensure all TSR nodes have a summary node
        

  logger.debug("updated store from version %d to %d", version, STORE_VERSION)
  root._v_attrs['version'] = STORE_VERSION
  fileh.flush()

def getnode(node, key):
  if hasattr(node, key):
    return getattr(node, key)
  else:
    raise NoData

class Stored(object):
  def __init__(self, store, node):
    self.store = store
    self.node = node

  @property
  def uuid(self):
    return self.node._v_name

  @property
  def metadata(self):
    return get_metadata(self.node)
  
  def __repr__(self):
    return "<%s on '%s'>" % (str(self.__class__.__name__), str(self.node))

class SpaceProxy(object):
  def get_space(self, name):
    try:
      return self.store.get_Space(self.metadata[name])
    except NoData:
      return None
    
  @property
  def instance_space(self):
    return self.get_space('instance_space')

  @property
  def class_space(self):
    return self.get_space('class_space')

  @property
  def feature_space(self):
    # Obtaining a feature space is tricky. We first need to determine if this space is native, or
    # if it has been transformed at some point. Even for native spaces, we need to handle compound
    # spaces.
    raise NotImplementedError

class StoredTaskSet(SpaceProxy, Stored, TaskSet):
  @property
  def classlabels(self):
    return self.store.get_Space(self.metadata['class_space'])

  @property
  def featurelabels(self):
    feature_spaces = self.metadata['feature_desc']
    labels = []
    for feature_space in sorted(feature_spaces):
      labels.extend(self.store.get_Space(feature_space))
    return labels
    
  def __len__(self):
    return len(self.node._v_groups)

  def __getitem__(self, key):
    g = self.node._v_groups
    task_node = None
    for node in g.values():
      if node._v_attrs.index == key:
        task_node = node
    if task_node is None:
      raise IndexError(key)
    return StoredTask(self.store, task_node)

class StoredTask(Stored, Task):
  @property
  def train_vectors(self):
    return Store._read_sparse_node(self.node.train_vectors)

  @property
  def train_classes(self):
    return self.node.train_classes.read()

  @property
  def train_indices(self):
    return self.node.train_indices.read()

  @property
  def train_sequence(self):
    if hasattr(self.node, 'train_sequence'):
      return Store._read_sparse_node(self.node.train_sequence)
    else:
      return None

  @property
  def test_vectors(self):
    return Store._read_sparse_node(self.node.test_vectors)

  @property
  def test_classes(self):
    return self.node.test_classes.read()

  @property
  def test_indices(self):
    return self.node.test_indices.read()

  @property
  def test_sequence(self):
    if hasattr(self.node, 'test_sequence'):
      return Store._read_sparse_node(self.node.test_sequence)
    else:
      return None

  @property
  def weights(self):
    return StoredWeights(self.node.weights)

class NodeProxy(UserDict.DictMixin):
  def __init__(self, node):
    self.node = node

  def __contains__(self, key):
    return key in self.node

  def keys(self):
    return list(self.node._v_children)


class StoredSummaries(NodeProxy):
  def __init__(self, node, overwrite = False):
    NodeProxy.__init__(self, node)
    self.overwrite = overwrite

  def __getitem__(self, key):
    if key in self.node:
      attr_node = getattr(self.node, key)._v_attrs
      return dict( (k,attr_node[k]) for k in attr_node._v_attrnamesuser )
    else:
      return {}

  def __setitem__(self, key, value):
    if key in self:
      summary_node = getattr(self.node, key)
      if not self.overwrite:
        # Check for key collisions first 
        old_keys = set(summary_node._v_attrs._v_attrnamesuser)
        new_keys = set(value.keys())
        overlap = old_keys & new_keys
        if len(overlap) != 0:
          raise ValueError, "Already had the following keys: %s" % str(list(overlap))
    else:
      summary_node = self.node._v_file.createGroup(self.node, key)
    for k, v in value.iteritems():
      summary_node._v_attrs[k] = v

  def __delitem__(self, key):
    if key in self:
      summary_node = getattr(self.node, key)
      self.node._v_file.removeNode(summary_node)
      return True
    else:
      return False

class StoredWeights(NodeProxy):
  def __getitem__(self, key):
    if key in self.node:
      return getattr(self.node, key).read()
    else:
      return None

  def __setitem__(self, key, value):
    if key in self.node:
      # TODO: Work out if we need to do something fancy with updates
      self.node._v_file.removeNode(getattr(self.node,key))
    # TODO: Work out if we can handle sparse arrays better
    self.node._v_file.createArray(self.node, key, value)

  def __delitem__(self, key):
    # TODO: Implement deletion
    raise NotImplementedError
  
class StoredResult(Stored, Result):
  @property
  def goldstandard(self):
    return self.node.goldstandard.read()

  @property
  def classifications(self):
    return self.node.classifications.read()

  @property
  def instance_indices(self):
    return self.node.instance_indices.read()

    
class StoredTaskSetResult(SpaceProxy, Stored, TaskSetResult):
  @property
  def taskset(self):
    # NOTE: This makes the assumption that the only difference between a tsr and the
    # corresponding taskset is the learner + learner_params.
    md = dict(self.metadata)
    del md['learner']
    del md['learner_params']
    try:
      return self.store.get_TaskSet(md)
    except NoData:
      raise NoData, "no corresponding taskset"

  @property
  def results(self):
    results = []
    for node in self.node._v_groups.values():
      if node._v_name != 'summary':
        results.append(StoredResult(self.store, node))

    try:
      results.sort(key=lambda r:r.metadata['index'])
    except KeyError:
      logger.warning("Tasks do not have index- returning in unspecified order")
    return results

  @property
  def summaries(self):
    # TODO: creation of summary nodes should happen at TSR initialization
    # This ad-hoc fix creates problems with read-only mode. We patch this
    # with an ad-hoc fix until we get round to ensuring all TSR have summary
    # nodes
    try:
      if not hasattr(self.node, 'summary'):
        self.node._v_file.createGroup(self.node,'summary')
    except tables.FileModeError:
      from collections import defaultdict
      return defaultdict(dict)
    return StoredSummaries(self.node.summary)

  def summarize(self, summary_fn, interpreter, force=False):
    if force:
      missing_keys = set(summary_fn.keys)
    else:
      summary = self.summaries[interpreter.__name__]
      missing_keys = set(summary_fn.keys) - set(summary)
    if len(missing_keys) > 0:
      summary_fn.init(self, interpreter)
      new_values = dict( (key, summary_fn[key]) for key in missing_keys )
      summaries = self.summaries
      summaries.overwrite = force
      summaries[interpreter.__name__] = new_values
    return self.summaries[interpreter.__name__]


class Store(object):
  """
  This is the master store class for hydrat. It manages all of the movement of data
  to and from disk.
  The fallback argument can be used to provide references to additional stores to
  attempt to read from, if the data requested is not present in this store.
  """
  def __init__(self, path, mode='r', fallback=None, recursive_close=True):
    """
    The store object has four major nodes:
    # spaces
    # datasets
    # tasksets
    # results
    """
    self.path = path
    self.recursive_close = recursive_close
    logger.debug("Opening Store at '%s', mode '%s'", self.path, mode)
    self.filelock = FileLock(self.path, timeout=0)
    # TODO: Extend filelock so it stores the pid of the locking process,
    #       and implement a lock-breaking mechanism that is aware of this.

    # The locking behaviour we need is a bit odd. We can open for reading mutliple 
    # times, and we can open for reading a file that is being appended to, but we 
    # cannot append to a file that is open for reading. We implement this behaviour
    # by always trying to lock. If we are unable to, we allow reads to proceed but
    # block any appends.

    try:
      self.filelock.acquire()
    except FileLockException:
      # just ignore the lock if we only want to read
      if mode in 'wa':
        raise StoreError("Store at '{0}' is locked".format(self.path))
      
    self.fileh = tables.openFile(self.path, mode=mode)
    self.mode = mode

    try:
      self._check_writeable()
      update_h5store(self.fileh)
    except IOError:
      pass

    self.root = self.fileh.root
    self.datasets = self.root.datasets
    self.spaces = self.root.spaces
    self.tasksets = self.root.tasksets
    self.results = self.root.results

    if fallback is None:
      # Open a 'null' store, which behaves like a store that contains no data
      self.fallback = NullStore()
    else:
      if not isinstance(fallback, Store):
        # TODO: allow just a path to be specified
        raise ValueError, "fallback argument must be a Store instance"
      self.fallback = fallback

    if not 'version' in self.root._v_attrs or self.root._v_attrs['version'] != STORE_VERSION:
      raise ValueError, "Store format is outdated; please open the store as writeable to automatically update it"

    # http://docs.python.org/reference/datamodel.html states that:
    # It is not guaranteed that __del__() methods are called for objects that still exist when the interpreter exits.
    #
    # We need to ensure that __del__ is called as it is used to release the file lock, so we
    # use an exitfunc to ensure this happens.
    import atexit; atexit.register(lambda: self.__del__())

  @classmethod
  def from_caller(cls, fallback=None):
    """
    Initialize a store, using the top-level calling module's name as the basis for the store filename.
    """
    # Open a store named after the top-level calling file
    import inspect
    stack = inspect.stack()
    filename = os.path.basename(stack[-1][1])
    store_path = config.getpath('paths','store') 
    path = os.path.join(store_path, os.path.splitext(filename)[0]+'.h5')
    logger.debug("from_caller: %s", path)
    return cls(path, 'a', fallback=fallback)
  
  def __str__(self):
    return "<Store mode '%s' @ '%s'>" % (self.mode, self.path)

  def close(self):
    if self.fallback is not None and self.recursive_close:
      self.fallback.close()
    if hasattr(self, 'fileh'):
      self.fileh.close()
    self.filelock.release()

  def __del__(self):
    self.close()

  ###
  # Utility Methods
  ###
  def _check_writeable(self):
    if self.mode not in "wa":
      raise IOError, "Store is not writeable!"

  @staticmethod
  def _read_sparse_node(node, shape=None):
    """
    We allow the shape to be overloaded so that we can accomodate
    feature maps where the underlying feature space has grown
    """
    dtype = node._v_attrs.dtype
    if shape is None: shape = node._v_attrs.shape
    n_ent = node._v_attrs.NROWS
    logger.debug("reading sparse node {0}({1} entries)".format(shape, n_ent))
    with Timer() as t:
      ax0 = node.read(field='ax0')
      ax1 = node.read(field='ax1')
      values = node.read(field='value')
      # TODO: This is turning out to be a blocker as it
      #       basically requires double memory to do the
      #       conversion. The options at this point are:
      #       1) change how sparse nodes are stored, so we
      #          can read back the sparse matrix without
      #          conversion
      #       2) use a disk-backed data structure somewhere
      m = csr_matrix((values,(ax0,ax1)), shape=shape)
    logger.debug("reading took {0:.1f}s ({1} entries/s)".format(t.elapsed, n_ent/t.elapsed))
    return m

  def _add_sparse_node( self
                      , where
                      , name
                      , type
                      , data
                      , title=''
                      , filters=None
                      ):
    node = self.fileh.createTable( where 
                                 , name
                                 , type
                                 , title
                                 , filters
                                 , expectedrows = data.nnz
                                 )
    attrs = node._v_attrs
    setattr(attrs, 'dtype', data.dtype)
    setattr(attrs, 'shape', data.shape)
    # Add the features to the table
    inst, feat = data.nonzero()
    logger.debug("writing sparse node {0}({1} entries)".format(data.shape, data.nnz))
    CHUNKSIZE = 1000000
    inst_i = iter(inst)
    feat_i = iter(feat)
    data_i = iter(data.data)
    t = Timer()
    with t:
      while True:
        inst_c = numpy.fromiter(itertools.islice(inst_i, CHUNKSIZE), 'uint64')
        feat_c = numpy.fromiter(itertools.islice(feat_i, CHUNKSIZE), 'uint64')
        data_c = numpy.fromiter(itertools.islice(data_i, CHUNKSIZE), data.dtype)
        if len(inst_c) == 0:
          break
        node.append(numpy.rec.fromarrays((inst_c, feat_c, data_c)))
      self.fileh.flush()
    logger.debug("writing took {0:.1f}s ({1} entries/s)".format(t.elapsed,data.nnz/t.elapsed))


  def has_Space(self, name):
    return (hasattr(self.spaces, name) or self.fallback.has_Space(name))
  
  ###
  # Add
  ###
  def add_Space(self, labels, metadata):
    """
    Add a space to the store. metadata must contain at
    least 'name' and 'type'.
    """
    self._check_writeable()
    if 'type' not in metadata:
      raise InsufficientMetadata, "metadata must contain type"
    if 'name' not in metadata:
      raise InsufficientMetadata, "metadata must contain name"
    if hasattr(self.spaces, metadata['name']):
      raise AlreadyHaveData, "Already have space %s" % metadata

    logger.debug( "Adding a %s space '%s' of %d Features"
                    , metadata['type']
                    , metadata['name']
                    , len(labels)
                    )

    # TODO: Maybe check the metadata for encoding?
    # Weak assumption that if the first label is unicode,
    # they all are unicode
    if isinstance(labels[0], unicode):
      encoding = 'utf8'
      # We encode all labels because PyTables rejects unicode
      labels = [ l.encode(encoding) for l in labels ]
    else:
      encoding = 'ascii'

    metadata['encoding'] = encoding

    # NOTE:
    # We set a default label representation. This is needed because of issues
    # in numpy with storing strings terminating in a null byte. The null byte
    # gets truncated, which prevents proper round-trip behaviour. We thus
    # have to encode labels before storing them, and decode them when retireving.
    metadata['repr'] = 'base64'
      
    # Check that labels are unique
    if len(labels) != len(set(labels)):
      raise ValueError, "labels are not unique!"

    repr_enc = metadata['repr']
    labels = [l.encode(repr_enc) for l in labels]
    new_space = self.fileh.createArray(self.spaces, metadata['name'], labels)

    for key in metadata:
      setattr(new_space.attrs, key, metadata[key])
    new_space.attrs.size = len(labels)

  def extend_Space(self, space_name, labels):
    # We do this by checking that the new space is a superset of the
    # old space, with the labels in the same order, then we delete the old 
    # space and add a new space.
    space = self.get_Space(space_name)
    if any( old != new for old, new in zip(space, labels)):
      raise StoreError, "New labels are not an extension of old labels"

    if len(labels) < len(space):
      raise StoreError, "New labels are less than old labels"

    if len(labels) == len(space):
      logger.debug("Space has not changed, no need to extend")
      return

    logger.debug("Extending '%s' from %d to %d features", space_name, len(space), len(labels))
    metadata = self.get_SpaceMetadata(space_name)

    # Set default label representation
    if 'repr' not in metadata:
      metadata['repr'] = 'base64'

    encoding = metadata['encoding']
    if encoding != 'ascii':
      labels = [l.encode(encoding) for l in labels]

    # Check that labels are unique
    if len(labels) != len(set(labels)):
      raise ValueError, "labels are not unique!"

    # Delete the old node
    try:
      self.fileh.removeNode(getattr(self.spaces, space_name))
    except tables.NoSuchNodeError:
      # If we cannot find this node, it probably is from a fallback
      # store. We should be able to safely ignore this and proceed to
      # writing the new space to the outer store.
      pass

    # Create the new node
    labels = [l.encode(metadata['repr']) for l in labels]
    new_space = self.fileh.createArray(self.spaces, space_name, labels)

    # Transfer the metadata
    for key in metadata:
      setattr(new_space.attrs, key, metadata[key])

    # Update the size
    new_space.attrs.size = len(labels)
    self.fileh.flush()


  def add_Dataset(self, name, instance_space, instance_ids):
    self._check_writeable()
    logger.debug("Adding dataset '%s'", name)
    if hasattr(self.datasets, name):
      raise AlreadyHaveData, "Already have dataset by name %s", name

    # Create a group for the DataSet
    ds = self.fileh.createGroup( self.fileh.root.datasets
                               , name
                               )

    attrs = ds._v_attrs

    # Note down our metadata
    attrs.name              = name
    attrs.instance_space    = instance_space
    attrs.num_instances     = len(instance_ids)

    # Create the instance_id array
    if hasattr(self.spaces, instance_space):
      # Check that the spaces match
      space = self.get_Space(instance_space)
      if (space != numpy.array(instance_ids)).any():
        raise ValueError, "Instance identifiers don't match existing instance space"
    else:
      self.add_Space(instance_ids, {'type':'instance', 'name':instance_space})

    # Create a group for Feature Data 
    self.fileh.createGroup(ds, "feature_data")

    # Create a group for Class Data 
    self.fileh.createGroup(ds, "class_data")

    # Create a group for Token Streams
    self.fileh.createGroup(ds, "tokenstreams")

    # Create a group for Token Streams
    self.fileh.createGroup(ds, "sequence")

    # Create a group for Splits
    self.fileh.createGroup(ds, "split")

    # return the newly created ds node
    return ds

  def add_FeatureDict(self, dsname, space_name, feat_map):
    self._check_writeable()
    logger.debug("Adding feature map to dataset '%s' in space '%s'", dsname, space_name)
    ds = getnode(self.datasets, dsname)
    space = self.get_Space(space_name)

    group = self.fileh.createGroup( ds.feature_data
                                  , space_name
                                  , "Sparse Feature Map %s" % space_name
                                  )
    # TODO: is this very expensive?
    group._v_attrs.type  = 'int' if all(isinstance(i[2], (int,long)) for i in feat_map) else 'float'

    fm_node = self.fileh.createTable( group
                                    , 'feature_map'
                                    , IntFeature if group._v_attrs.type == 'int' else RealFeature 
                                    , title = 'Sparse Feature Map'
                                    , filters = tables.Filters(complevel=5, complib='zlib') 
                                    , expectedrows = len(feat_map)
                                    )

    # Initialize space to store instance sizes.
    n_inst = ds._v_attrs['num_instances']
    
    attrs = fm_node._v_attrs
    setattr(attrs, 'dtype', group._v_attrs.type)
    setattr(attrs, 'shape', (n_inst, len(space)))

    logger.debug("  writing table")
    # Add the features to the table
    feature = fm_node.row
    #for i, j, v in ProgressIter(feat_map, label='write table'):
    for i, j, v in feat_map:
      feature['ax0'] = i
      feature['ax1'] = j
      feature['value'] = v
      feature.append()

    self.fileh.flush()

  def add_FeatureMap(self, dsname, space_name, feat_map):
    self._check_writeable()
    logger.debug("Adding feature map to dataset '%s' in space '%s'", dsname, space_name)
    ds = getnode(self.datasets, dsname)
    space = self.get_Space(space_name)
    if feat_map.shape[1] != len(space):
      raise ValueError, "feature map is the wrong shape for this feature space"

    group = self.fileh.createGroup( ds.feature_data
                                  , space_name
                                  , "Sparse Feature Map %s" % space_name
                                  )
    group._v_attrs.type  = int if issubclass(feat_map.dtype.type, numpy.int) else float

    _typ = IntFeature if issubclass(feat_map.dtype.type,numpy.int) else RealFeature 
    fm_node = self._add_sparse_node\
                  ( group
                  , 'feature_map'
                  , _typ 
                  , feat_map
                  , filters = tables.Filters(complevel=5, complib='zlib') 
                  )
     
    self.fileh.flush()


  def add_ClassMap(self, dsname, space_name, class_map):
    self._check_writeable()
    logger.debug("Adding Class Map to dataset '%s' in space '%s'", dsname, space_name)
    ds = getnode(self.datasets, dsname)
    space = self.get_Space(space_name)

    num_inst = self.get_SpaceMetadata(ds._v_attrs['instance_space'])['size'] 
    num_classes = len(space)

    # Check that the map is of the right shape for the dataset and space
    if class_map.shape != (num_inst, num_classes):
      raise StoreError, "Wrong shape for class map!"

    group =  self.fileh.createGroup( ds.class_data
                                   , space_name
                                   , "Data for %s" % space_name
                                   )

    cm_node = self.fileh.createCArray( group
                                     , 'class_map'
                                     , tables.BoolAtom()
                                     , class_map.shape
                                     , title = 'Class Map'
                                     , filters = tables.Filters(complevel=5, complib='zlib') 
                                     )
    cm_node[:] = class_map
    cm_node.flush()
                               
  def add_TaskSet(self, taskset):
    # TODO: Find some way to make this atomic! Otherwise, we can write incomplete tasksets!!
    #       The best way to do this is probably to introduce a top-level 'pending'
    #       node. A hardlink to the new node will be attached when we create the new
    #       node but before we start writing to it. This hardlink is removed when
    #       writing is complete. Thus, if we open up a store and find anything attached
    #       to the pending node, we know that the item is incomplete and should be
    #       deleted.
    self._check_writeable()
    
    # Copy the metadata as we are not allowed to modify it directly
    md = dict(taskset.metadata)
    taskset_uuid = uuid.uuid4()

    taskset_entry_tag = str(taskset_uuid)
    taskset_entry = self.fileh.createGroup(self.tasksets, taskset_entry_tag)
    taskset_entry_attrs = taskset_entry._v_attrs

    for key in md:
      setattr(taskset_entry_attrs, key, md[key])

    try:
      logger.debug('Adding a taskset %s', str(md))

      for i,task in enumerate(ProgressIter(taskset, label="Adding Tasks")):
        logger.debug('  adding task {0}'.format(i))
        self._add_Task(task, taskset_entry)
      self.fileh.flush()
    except Exception, e:
      # Delete the node if any exceptions occur
      logger.error("{0}: {1}".format(type(e),e))
      self.fileh.removeNode(taskset_entry, recursive=True)
      raise

    return taskset_entry_tag

  ###
  # List
  ###
  def list_Spaces(self, space_type, dsname = None):
    if dsname is None:
      retval = set(s._v_name for s in self.spaces if s._v_attrs.type == space_type)
    else:
      try:
        ds = getnode(self.datasets, dsname)
        if space_type == 'feature':
          retval = set(node._v_name for node in ds.feature_data)
        elif space_type == 'class':
          retval = set(node._v_name for node in ds.class_data)
        else:
          raise ValueError, "don't know about space type %s" % space_type
      except NoData:
        retval = set()
    retval |= self.fallback.list_Spaces(space_type, dsname)
    return retval

  def list_ClassSpaces(self, dsname = None):
    return self.list_Spaces('class', dsname=dsname)

  def list_FeatureSpaces(self, dsname = None):
    return self.list_Spaces('feature', dsname=dsname)

  def list_InstanceSpaces(self):
    return self.list_Spaces('instance')

  def list_Datasets(self):
    retval = set( ds._v_attrs.name for ds in self.datasets)
    retval |= self.fallback.list_Datasets()
    return retval

  ###
  # Get
  ###
  def get_Metadata(self, parent, identifier):
    try:
      node = getnode(getattr(self, parent), identifier)
      metadata = dict(   (key, getattr(node._v_attrs,key)) 
                    for  key 
                    in   node._v_attrs._v_attrnamesuser
                    )
      return metadata
    except (AttributeError, NoData):
      return self.fallback.get_Metadata(parent, identifier)

  def get_SpaceMetadata(self, space_name):
    """
    @param space_name: Identifier of the relevant space
    @type tag: string
    @rtype: dict of metadata key-value pairs
    """
    return self.get_Metadata('spaces', space_name)

  def get_DatasetMetadata(self, dsname):
    """
    @param dsname: Identifier of the relevant dataset
    @type dsname: string
    @rtype: dict of metadata key-value pairs
    """
    return self.get_Metadata('datasets', dsname)

  def get_Space(self, space_name):
    """
    @param space_name: Name of the space 
    @rtype: pytables array
    """
    try:
      space = getnode(self.spaces, space_name)
    except NoData:
      if self.fallback is not None:
        return self.fallback.get_Space(space_name)
      else:
        raise NoData, "Store does not have space '%s'" % space_name
    metadata = self.get_SpaceMetadata(space_name)
    data = space.read()
    if 'repr' in metadata:
      # Undo any represenation-level encoding
      repr_enc = metadata['repr']
      data = [ d.decode(repr_enc) for d in data ]
    try:
      encoding = metadata['encoding']
    except KeyError:
      logger.warning('Space %s does not have encoding data!', tag)
      encoding = 'nil'
    if encoding != 'nil' and encoding != 'ascii':
      data = [ d.decode(encoding) for d in data ]
    if len(space) != len(set(space)):
      raise ValueError, "space is not unique!"
    return data

  # Deprecated upon introduction of instance spaces. use get_Space instead.
  @deprecated
  def get_InstanceIds(self, dsname):
    """
    @param dsname: Dataset name
    @return: Instance identifiers for dataset
    @rtype: List of strings
    """
    return self.get_Space(dsname)

  def has_Data(self, dsname, space_name):
    try:
      ds = getnode(self.datasets, dsname)
      return (  hasattr(ds.class_data,   space_name) 
            or hasattr(ds.feature_data, space_name)
            or self.fallback.has_Data(dsname, space_name)
            )
    except NoData:
      return self.fallback.has_Data(dsname, space_name)

  def has_Dataset(self, dsname):
    # TODO: Worry about consistency issues as a result of
    # writing to a dataset that only exists in the fallback
    if hasattr(self.datasets, dsname):
      return True
    else:
      return self.fallback.has_Dataset(dsname)

  def get_ClassMap(self, dsname, space_name):
    """
    @param dsname: Name of the dataset
    @param space_name: Name of the class space
    @return: data corresponding to the given dataset in the given class space
    @rtype: pytables array
    """
    try:
      ds = getnode(self.datasets, dsname)
      class_node = getnode(ds.class_data, space_name) 
    except NoData:
      return self.fallback.get_ClassMap(dsname, space_name)

    data = getattr(class_node, 'class_map')
    metadata = dict\
                 ( dataset=dsname
                 , class_space=space_name
                 , instance_space=ds._v_attrs.instance_space
                 )
    return ClassMap(data.read(), metadata=metadata)

  def get_FeatureMap(self, dsname, space_name):
    """
    @param dsname: Name of the dataset
    @param space_name: Name of the feature space
    @return: data corresponding to the given dataset in the given feature space
    @rtype: varies 
    """
    #import inspect
    #caller = inspect.stack()[1]
    #print "get_FeatureMap({0},{1}) called by {caller[3]}({caller[1]}:{caller[2]})".format(dsname, space_name, caller=caller)
    try:
      ds = getnode(self.datasets, dsname)
      feature_node = getnode(ds.feature_data, space_name)
    except NoData:
      return self.fallback.get_FeatureMap(dsname, space_name)

    data_type = feature_node._v_attrs.type
    logger.debug("returning {0}:{1} ({2})".format(dsname, space_name, data_type))
    fm = getattr(feature_node, 'feature_map')
    n_inst = self.get_SpaceMetadata(ds._v_attrs['instance_space'])['size'] 
    n_feat = self.get_SpaceMetadata(space_name)['size'] 
    m = self._read_sparse_node(fm,shape=(n_inst, n_feat))
    metadata = dict\
                 ( dataset=dsname
                 , feature_desc=(space_name,)
                 , instance_space=ds._v_attrs.instance_space
                 )
    return FeatureMap(m, metadata=metadata) 

  def new_TaskSet(self, taskset):
    """Method which checks if a TaskSet is already in the Store. It will not
    access the tasks unless the source metadata is not yet in the Store.
    This allows for on-demand generation of tasks via property-based tasks.
    """
    if not self.has_TaskSet(taskset.metadata): 
      self.add_TaskSet(taskset)
    return self.get_TaskSet(taskset.metadata)

  def _add_Task(self, task, ts_entry): 
    #TODO: Enforce type of train/test indices to be integer sequences
    self._check_writeable()

    md = dict(task.metadata)

    task_uuid = uuid.uuid4()
    task_tag = str(task_uuid)

    # Create a group for the task
    task_entry = self.fileh.createGroup(ts_entry, task_tag)
    task_entry_attrs = task_entry._v_attrs

    # Add the metadata
    for key in md:
      setattr(task_entry_attrs, key, md[key])

    # Add the class matrices 
    # TODO: Current implementation has the side effect of expanding all tasks,
    #       meaning we lose the memory savings of an InMemoryTask. Maybe this
    #       is not a big deal, but if it is we need to look into how to handle
    #       it.
    self.fileh.createArray(task_entry, 'train_classes', task.train_classes)
    self.fileh.createArray(task_entry, 'test_classes', task.test_classes)
    self.fileh.createArray(task_entry, 'train_indices', task.train_indices)
    self.fileh.createArray(task_entry, 'test_indices', task.test_indices)
    tr = task.train_vectors
    te = task.test_vectors

    self._add_sparse_node( task_entry
                         , 'train_vectors'
                         , IntFeature if issubclass(tr.dtype.type,numpy.integer) else RealFeature 
                         , tr
                         , filters = tables.Filters(complevel=5, complib='zlib') 
                         )
    self._add_sparse_node( task_entry
                         , 'test_vectors'
                         , IntFeature if issubclass(te.dtype.type,numpy.integer) else RealFeature 
                         , te 
                         , filters = tables.Filters(complevel=5, complib='zlib') 
                         )

    sqr = task.train_sequence
    sqe = task.test_sequence

    if sqr is not None:
      self._add_sparse_node( task_entry
                           , 'train_sequence'
                           , BoolFeature
                           , sqr
                           , filters = tables.Filters(complevel=5, complib='zlib') 
                           )

    if sqe is not None:
      self._add_sparse_node( task_entry
                           , 'test_sequence'
                           , BoolFeature
                           , sqe
                           , filters = tables.Filters(complevel=5, complib='zlib') 
                           )
    weights_node = self.fileh.createGroup(task_entry, 'weights')
    for key in task.weights:
      new_weight = self.fileh.createArray\
                      ( weights_node
                      , key
                      , task.weights[key]
                      )

    self.fileh.flush()

  def extend_Weights(self, taskset):
    # TODO: Do we need to perform a check for some kind of characteristic
    #       of the weight?
    taskset_entry  = getnode(self.tasksets, str(taskset.metadata['uuid']))
    for task in taskset.tasks:
      task_tag = str(task.metadata['uuid'])
      task_entry     = getattr(taskset_entry, task_tag)
      for key in task.weights:
        if not hasattr(task_entry.weights, key):
          new_weight = self.fileh.createArray\
                          ( task_entry.weights
                          , key
                          , task.weights[key]
                          )
    self.fileh.flush()
                
  def has_TaskSet(self, desired_metadata):
    """ Check if any taskset matches the specified metadata """
    return bool(self._resolve_TaskSet(desired_metadata)) or self.fallback.has_TaskSet(desired_metadata)

  def get_TaskSet(self, desired_metadata):
    """ Convenience function to ensure exactly one TaskSet is returned"""
    tasksets = self.get_TaskSets(desired_metadata)
    if len(tasksets) == 0: 
      raise NoData
    elif len(tasksets) > 1: 
      raise InsufficientMetadata
    return tasksets[0]

  def get_TaskSets(self, desired_metadata):
    # NOTE: When working with tasksets from the fallback, it is possible that the
    # primary is writeable but the fallback is not, so writing weights can be 
    # an issue. The ideal solution would be a copy-on-write of the entire
    # taskset, but it is a non-trivial task to implement and as such implementation
    # is deferred until it is shown to be needed.
    tags = self._resolve_TaskSet(desired_metadata)
    tasksets = []
    for tag in tags:
      try:
        taskset = self._get_TaskSet(tag)
      except tables.NoSuchNodeError:
        logger.warning('Removing damaged TaskSet node with metadata %s', str(desired_metadata))
        continue
      tasksets.append(taskset)

    try:
      fallback_tasks = self.fallback.get_TaskSets(desired_metadata)
    except NoData:
      fallback_tasks = []

    tasksets.extend(fallback_tasks)
    return tasksets
  
  def _del_TaskSet(self, taskset_tag):
    if not hasattr(self.tasksets, taskset_tag):
      raise KeyError, str(taskset_tag)
    self.fileh.removeNode(self.tasksets, taskset_tag, True)

  def _get_TaskSet(self, taskset_tag, weights = None):
    try:
      taskset_entry  = getnode(self.tasksets, taskset_tag)
    except AttributeError:
      raise KeyError, str(taskset_tag)
    return StoredTaskSet(self, taskset_entry)

  def _resolve_TaskSet(self, desired_metadata):
    """Returns all tags whose entries match the supplied metadata"""
    desired_keys = []
    for node in self.tasksets._v_groups:
      attrs = getnode(self.tasksets,node)._v_attrs
      if metadata_matches(attrs, desired_metadata):
        desired_keys.append(node)
    return desired_keys

  def new_TaskSetResult(self, tsr):
    """Method which checks if a TaskSetResult is already in the Store. It will not
    access the results unless the source metadata is not yet in the Store.
    This allows for on-demand generation of results via property-based results.
    """
    if not self.has_TaskSetResult(tsr.metadata): 
      self.add_TaskSetResult(tsr)
    return self.get_TaskSetResult(tsr.metadata)

  def has_TaskSetResult(self, desired_metadata):
    """ Check if the TaskSetResult already exists """
    return bool(self._resolve_TaskSetResults(desired_metadata)) or self.fallback.has_TaskSetResult(desired_metadata)

  def get_TaskSetResult(self, desired_metadata):
    """ Convenience function to bypass tag resolution """
    tags = self._resolve_TaskSetResults(desired_metadata)
    if len(tags) == 0: 
      return self.fallback.get_TaskSetResult(desired_metadata)
    elif len(tags) > 1: 
      raise InsufficientMetadata
    return self._get_TaskSetResult(tags[0])

  def get_TaskSetResults(self, desired_metadata):
    tags = self._resolve_TaskSetResults(desired_metadata)
    try:
      fallback_tasks = self.fallback.get_TaskSetResults(desired_metadata)
    except NoData:
      fallback_tasks = []
    return [self._get_TaskSetResult(t) for t in tags] + fallback_tasks
  
  def _del_TaskSetResult(self, tsr_tag):
    if not hasattr(self.results, tsr_tag):
      raise KeyError, str(tsr_tag)
    self.fileh.removeNode(self.results, tsr_tag, True)

  def _get_TaskSetResult(self, tsr_tag):
    try:
      tsr_entry  = getnode(self.results, tsr_tag)
    except AttributeError:
      raise KeyError, str(tsr_tag)
    return StoredTaskSetResult(self, tsr_entry)

  def _resolve_TaskSetResults(self, desired_metadata):
    """Returns all tags whose entries match the supplied metadata"""
    keys = self.results._v_groups.keys()
    desired_keys = []
    for key in keys:
      attrs = getattr(self.results,key)._v_attrs
      if metadata_matches(attrs, desired_metadata):
        desired_keys.append(key)
    return desired_keys

  def add_TaskSetResult(self, tsr, additional_metadata={}):
    self._check_writeable()

    # Evaluate this to ensure any errors in results generation are caught early
    results = tsr.results

    md = dict(tsr.metadata)
    md.update(additional_metadata)

    tsr_uuid = uuid.uuid4()

    tsr_entry_tag = str(tsr_uuid)
    tsr_entry = self.fileh.createGroup(self.results, tsr_entry_tag)
    tsr_entry_attrs = tsr_entry._v_attrs

    for key in md:
      setattr(tsr_entry_attrs, key, md[key])

    for i,result in enumerate(results):
      self._add_Result(result, tsr_entry, dict(index=i))
    self.fileh.flush()
    return tsr_entry_tag

  def _add_Result(self, result, tsr_entry, additional_metadata={}):
    self._check_writeable()
    result_uuid = uuid.uuid4()
    result_tag = str(result_uuid)

    # Create a group for the result
    result_entry = self.fileh.createGroup(tsr_entry, result_tag)
    result_entry_attrs = result_entry._v_attrs

    # Add the metadata
    for key in result.metadata:
      setattr(result_entry_attrs, key, result.metadata[key])
    for key in additional_metadata:
      setattr(result_entry_attrs, key, additional_metadata[key])

    # Add the class matrices 
    self.fileh.createArray(result_entry, 'classifications', result.classifications)
    self.fileh.createArray(result_entry, 'goldstandard', result.goldstandard)
    self.fileh.createArray(result_entry, 'instance_indices', result.instance_indices)
     
  ###
  # TokenStream
  ###

  def has_TokenStreams(self, dsname, stream_name):
    dsnode = getnode(self.datasets, dsname)
    return hasattr(dsnode.tokenstreams, stream_name)

  def add_TokenStreams(self, dsname, stream_name, tokenstreams):
    # TODO: Handle trying to create a stream that already exists
    dsnode = getnode(self.datasets, dsname)
    stream_array = self.fileh.createVLArray( dsnode.tokenstreams
                                           , stream_name
                                           , tables.ObjectAtom()
                                           , filters = tables.Filters(complevel=5, complib='zlib') 
                                           )
    for stream in ProgressIter(tokenstreams, label="Adding TokenStreams '%s'" % stream_name):
      # TODO: don't allow lists of lists. For some reason storing lists is
      # substantially slower than storing tuples or strings. Looks to be a
      # cPickle issue.
      stream_array.append(stream)

  def get_TokenStreams(self, dsname, stream_name):
    try:
      dsnode = getnode(self.datasets, dsname)
      tsnode = getnode(dsnode.tokenstreams, stream_name)
      return iter(tsnode)
    except NoData:
      return self.fallback.get_TokenStreams(dsname, stream_name)

  def list_TokenStreams(self, dsname):
    try:
      dsnode = getnode(self.datasets, dsname)
      retval = set(node._v_name for node in dsnode.tokenstreams)
    except NoData:
      retval = set()
    retval |= self.fallback.list_TokenStreams(dsname)
    return retval

  ###
  # Sequence
  ###
  def add_Sequence(self, dsname, seq_name, sequence):
    # sequence should arrive as a boolean matrix. axis 0 is parent, axis 1 is child.
    if not issubclass(sequence.dtype.type, numpy.bool_):
      raise ValueError, "sequence must be a boolean matrix"
    if not sequence.shape[0] == sequence.shape[1]:
      raise ValueError, "sequence must be square"

    dsnode = getnode(self.datasets, dsname)
    self._add_sparse_node( dsnode.sequence
                         , seq_name
                         , BoolFeature 
                         , sequence
                         , filters = tables.Filters(complevel=5, complib='zlib') 
                         )

  def get_Sequence(self, dsname, seq_name):
    try:
      dsnode = getnode(self.datasets, dsname)
      sqnode = getnode(dsnode.sequence, seq_name)
      # Should be reading each row of the array as a member of a sequence
      # e.g. a row is a thread, each index is the instance index in dataset representing posts
      # returns a list of arrays.
      return self._read_sparse_node(sqnode)
    except (AttributeError,NoData):
      return self.fallback.get_Sequence(dsname, seq_name)

  def list_Sequence(self, dsname):
    if dsname in self.datasets:
      dsnode = getnode(self.datasets, dsname)
      retval = set(node._v_name for node in dsnode.sequence)
    else:
      retval = set()
    retval |= self.fallback.list_Sequence(dsname)
    return retval

  ###
  # Splits
  ###
  def list_Split(self, dsname):
    if dsname in self.datasets:
      dsnode = getnode(self.datasets, dsname)
      retval = set(node._v_name for node in dsnode.split)
    else:
      retval = set()
    retval |= self.fallback.list_Split(dsname)
    return retval

  def add_Split(self, dsname, split_name, split):
    # TODO: Sanity checks
    self._check_writeable()

    # Handle cases where the dsnode only exists in the fallback.
    try:
      dsnode = getnode(self.datasets, dsname)
    except NoData:
      md = self.get_DatasetMetadata(dsname)
      ids = self.get_Space(md['instance_space'])
      dsnode = self.add_Dataset(dsname, md['instance_space'], ids) 

    sp_node = self.fileh.createCArray( dsnode.split
                                     , split_name
                                     , tables.BoolAtom()
                                     , split.shape
                                     , title = split_name
                                     , filters = tables.Filters(complevel=5, complib='zlib') 
                                     )
    sp_node[:] = split
    sp_node.flush()

  def get_Split(self, dsname, split_name):
    try:
      dsnode = getnode(self.datasets, dsname)
      data = getnode(dsnode.split, split_name)
    except NoData:
      return self.fallback.get_Split(dsname, split_name)
    return data.read()

  ###
  # Merge
  ###
  def merge(self, other, allow_duplicate=False, do_spaces=True, do_datasets=True, do_tasksets=True, do_results=True):
    """
    Merge the other store's contents into self.
    We can copy tasksets and results verbatim, but spaces and datasets need to 
    take into account a possible re-ordering of features.
    Weights get another pass over tasksets
    TODO: Second pass on results to copy summaries
    """
    #TODO: May need to organize a staging area to ensure this merge is atomic
    if self.mode == 'r': raise ValueError, "Cannot merge into read-only store"
    ignored_md = ['uuid', 'avg_learn', 'avg_classify', 'name', 'feature_name', 'class_name']

    space_direct_copy = [] # Spaces we copy directly, meaning the featuremap can be copied too
    space_feature_mapping = {}
    if do_spaces or do_datasets:
      # Must do spaces if we do datasets, because spaces may have been updated
      for space_node in ProgressIter(list(other.spaces), label='Copying spaces'):
        logger.debug("Considering space '%s'", space_node._v_name)
        space_name = space_node._v_name
        if hasattr(self.spaces, space_name):
          logger.debug('Already had %s', space_name)
          src_space = other.get_Space(space_name)
          # Need to merge these. Feature spaces can be extended, but there is no mechanism for doing the same with class
          # spaces at the moment, so we must reject any that do not match. 
          dst_space = self.get_Space(space_name)
          if src_space == dst_space:
            logger.debug('  Exact match')
            space_direct_copy.append(space_name)
          else:
            md = get_metadata(space_node)
            if md['type'] == 'class':
              raise ValueError, "Cannot merge due to different versions of %s" % str(md)
            elif md['type'] == 'feature':
              logger.debug('  Attempting to merge %s', str(md))
              # Reconcile the spaces. 
              ## First we need to compute the new features to add
              new_feats = sorted(set(src_space) - set(dst_space))
              logger.debug('    Identified %d new features', len(new_feats))
              reconciled_space = dst_space + new_feats
              if len(new_feats) != 0:
                # Only need to extend if new features are found.
                self.extend_Space(space_name, reconciled_space)
              ## Now we need to build the mapping from the external space to ours
              space_index = dict( (k,v) for v,k in enumerate(reconciled_space))
              space_feature_mapping[space_name] = dict( (i,space_index[s]) for i,s in enumerate(src_space))
            else:
              raise ValueError, "Unknown type of space"
        else:
          self.fileh.copyNode(space_node, newparent=self.spaces)
          space_direct_copy.append(space_name)
        
    if do_datasets:
      for src_ds in ProgressIter(list(other.datasets), label='Copying datasets'):
        dsname = src_ds._v_name

        logger.debug("Considering dataset '%s'", dsname)
        if hasattr(self.datasets, dsname):
          logger.warning("already had dataset '%s'", dsname)
          dst_ds = getattr(self.datasets, dsname)
          # Failure to match instance_id is an immediate reject
          if dst_ds._v_attrs.instance_space != src_ds._v_attrs.instance_space:
            raise ValueError, "Instance identifiers don't match for dataset %s" % dsname
          # The hardest to handle is the feature data, since we may need to rearrange feature maps
        else:
          instance_space = other.get_DatasetMetadata(dsname)['instance_space']
          self.add_Dataset(dsname, instance_space, other.get_Space(dsname))
          dst_ds = getattr(self.datasets, dsname)

        node_names = ['class_data', 'sequence', 'tokenstreams']
        for name in node_names:
          logger.debug('Copying %s',name)
          if hasattr(src_ds, name):
            src_parent = getattr(src_ds, name)
            #TODO: may need to handle incomplete destination nodes
            dst_parent = getattr(dst_ds, name)
            for node in src_parent:
              if hasattr(dst_parent, node._v_name):
                logger.warning("already had '%s' in '%s'", node._v_name, name)
              else:
                self.fileh.copyNode(node, newparent=dst_parent, recursive=True)
          else:
            logger.warning("Source does not have '%s'", name)

        logger.debug('Copying feature_data')
        for node in src_ds.feature_data:
          space_name = node._v_name
          if hasattr(dst_ds.feature_data, space_name):
            logger.warning("already had '%s' in 'feature_data'", space_name) 
          elif space_name in space_direct_copy:
            # Direct copy the feature data because the destination store did not have this
            # space or had exactly this space
            logger.debug("direct copy of '%s' in 'feature_data'", space_name)
            self.fileh.copyNode(node, newparent=dst_ds.feature_data, recursive=True)
          else:
            ax0 = node.feature_map.read(field='ax0')
            ax1 = node.feature_map.read(field='ax1')
            value = node.feature_map.read(field='value')
            feature_mapping = space_feature_mapping[space_name]

            feat_map = [ (i,feature_mapping[j],v) for (i,j,v) in zip(ax0,ax1,value)]
            self.add_FeatureDict(dsname, space_name, feat_map)

      
    # TASKS & RESULTS
    def __merge(datum, check):
      logger.debug("Copying %s", datum)
      src_node = getattr(other, datum)
      dst_node = getattr(self, datum)
      for t in ProgressIter(list(src_node), label='Copying %s' % datum):
        logger.debug("Considering %s '%s'", datum, t._v_name)

        # Check if the exact result has been previously copied
        if t._v_name in dst_node:
          logger.warn("Skipping previous %s: %s", datum, t._v_name)
        else:
          md = get_metadata(t)
          for i in ignored_md: 
            if i in md: 
              del md[i]
          # Check for equivalent metadata
          if not allow_duplicate and check(md):
            logger.warn("Ignoring duplicate in %s: %s", datum, str(md))
          else:
            try:
              self.fileh.copyNode(t, newparent=dst_node, recursive=True)
            except tables.NoSuchNodeError:
              logger.critical("Damaged node skipped")

    if do_tasksets:
      # Copy entire nodes
      __merge('tasksets', self.has_TaskSet)
      # Now work our way through and check if any weights need updating
      for src in ProgressIter(other.get_TaskSets({}), label='Copying weights'):
        if src.node._v_name in self.tasksets:
          dst = StoredTaskSet(self, getattr(self.tasksets, src.node._v_name))
        else:
          md = dict(src.metadata)
          for i in ignored_md: 
            if i in md: 
              del md[i]
          dst = self.get_TaskSet(md)
        # sanity check for compatibility
        if len(src.tasks) != len(dst.tasks):
          logger.warning('number of tasks in src and dst do not match; skipping')
          continue
        for i, task in enumerate(src.tasks):
          dst.tasks[i].weights.update(src.tasks[i].weights)

    if do_results:
      __merge('results', self.has_TaskSetResult)

class NullStore(Store):
  def __init__(self):
    def null_list(*args, **kwargs): return set()
    def null_has(*args, **kwargs): return False
    def null_get(*args, **kwargs): raise NoData, "args: %s kwargs: %s" % (repr(args), repr(kwargs))
      
    for key in dir(self.__class__):
      if key.startswith('get_'):
        setattr(self, key, null_get)
      elif key.startswith('list_'):
        setattr(self, key, null_list)
      elif key.startswith('has_'):
        setattr(self, key, null_has)

  def close(self):
    pass
