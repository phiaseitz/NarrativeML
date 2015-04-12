"""
The Model API exposes all the data from a transformed representation
of an underlying dataset. The most basic objects are what are mostly
referred to as the doc_map, which is a 2d array of instances where axis 0 is
the instance and axis 1 are the feature values, and the class_map, where
axis 0 is the instance and axis 1 are the class membership values.
"""
class Model(object):
  pass

class ImmediateModel(Model):
  """
  Creates the most basic usable object to satisfy the Model API.
  Keeps everything in memory.
  """
  def __init__( self
              , feature_vectors
              , class_vectors = None
              , docids = None
              , classlabels = None
              , features = None
              , metadata = {}
              ):
    if class_vectors is not None:
      assert feature_vectors.shape[0] == class_vectors.shape[0]
    self.feature_vectors = feature_vectors
    self.class_vectors = class_vectors

    if docids is None:
      self.docids = [ 'd%04d' % i  for i in range(feature_vectors.shape[0])]
      metadata['dataset'] = 'immediate_model'
    else:
      self.docids = docids

    if classlabels is None:
      self.classlabels = [ 'c%02d' % i  for i in range(class_vectors.shape[1])]
      metadata['class_name'] = 'immediate_model'
    else:
      self.classlabels = classlabels

    if features is None:
      self.features = [ 'f%06d' % i  for i in range(feature_vectors.shape[1])]
      metadata['feature_name'] = 'immediate_model'
    else:
      self.features = features

    required_metadata = [ 'dataset'
                        , 'feature_name'   
                        , 'class_name'
                        ] 
    for m in required_metadata:
      if m not in metadata:
        raise ValueError, "Model metadata must contain %s" % m
    self.metadata = metadata.copy()

class ClassMap(object):
  def __init__(self, raw, metadata={}):
    self.raw = raw
    self.metadata = dict(metadata)

  def __getitem__(self, key):
    # TODO: Take note of what has been selecte somehow?
    return ClassMap(self.raw[key], self.metadata)

