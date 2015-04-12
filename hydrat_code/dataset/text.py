import os, gzip
from hydrat.dataset import Dataset
import hydrat.common.extractors as ext
from hydrat.common.filedict import FileDict

class TextDataset(Dataset):
  """ Base class for datasets where instances can be represented
      as single string. Ideal for traditional text classification
      tasks.

      The only requirement for subclassing TextDataset is that the 
      subclass must implement the ts_byte method, which returns a 
      dictionary mapping from the instance identifier to the
      text of the instance.
  """
  def ts_byte(self):
    """
    Return a dictionary from instance identifiers
    to the content of the instance in a string 
    This should be a normal byte string.
    """
    raise NotImplementedError

  def from_byte(self, fn):
    """
    Return a dictionary from instance identifiers to
    a bytestream after fn has been applied to the bytestream.
    """
    return self.ts2ts('byte', fn)

class SingleDir(TextDataset):
  """ Mixin for a dataset that has all of its source text files
  in a single directory. Requires that the deriving class
  implements a data_path method.
  """
  def data_path(self):
    raise NotImplementedError, "Deriving class must implement this"

  def ts_byte(self):
    path = self.data_path()
    instances = FileDict()
    for filename in os.listdir(path):
      filepath = os.path.join(path, filename)
      if os.path.isfile(filepath):
        instances[filename] = filepath
    return instances

class DirPerClass(TextDataset):
  def data_path(self):
    raise NotImplementedError, "Deriving class must implement this"

  def identifiers(self):
    path = self.data_path()
    ids = []
    cls = [ c for c in os.listdir(path) if os.path.isdir(os.path.join(path, c)) ]
    for cl in cls:
      for file in os.listdir(os.path.join(path,cl)):
        instance_id = '%s_%s'%(cl, file)
        ids.append(instance_id)
    return ids

  def ts_byte(self):
    path = self.data_path()
    ts = FileDict()
    cls = [ c for c in os.listdir(path) if os.path.isdir(os.path.join(path, c)) ]
    for cl in cls:
      for file in os.listdir(os.path.join(path,cl)):
        instance_id = '%s_%s'%(cl, file)
        ts[instance_id] = os.path.join(path, cl, file)
    return ts

  def cm_dirname(self):
    path = self.data_path()
    cm = {}
    cls = [ c for c in os.listdir(path) if os.path.isdir(os.path.join(path, c)) ]
    for cl in cls:
      for file in os.listdir(os.path.join(path,cl)):
        instance_id = '%s_%s'%(cl, file)
        cm[instance_id] = [ cl ]
    return cm 

  def dirname2class(self, mapping):
    cm_filename = self.classmap('dirname')
    retval = {}
    for key in cm_filename:
      retval[key] = [ mapping[v] for v in cm_filename[key] ]
    return retval

class FilePerClass(TextDataset):
  def data_path(self):
    raise NotImplementedError, "Deriving class must implement this"

  def ts_byte(self):
    path = self.data_path()
    ts = {}
    for cl in os.listdir(path):
      if cl.lower().endswith('.gz'):
        f = gzip.open(os.path.join(path, cl))
      else:
        f = open(os.path.join(path, cl))
      for i,instance in enumerate(f):
        instance_id = '%s%d'%(cl, i)
        ts[instance_id] = instance
      f.close()
    return ts

  def cm_filename(self):
    path = self.data_path()
    cm = {}
    for cl in os.listdir(path):
      if cl.lower().endswith('.gz'):
        f = gzip.open(os.path.join(path, cl))
      else:
        f = open(os.path.join(path, cl))
      for i,instance in enumerate(f):
        instance_id = '%s%d'%(cl, i)
        cm[instance_id] = [cl]
      f.close()
    return cm

  def filename2class(self, mapping):
    cm_filename = self.classmap('filename')
    retval = {}
    for key in cm_filename:
      retval[key] = [ mapping[v] for v in cm_filename[key] ]
    return retval


class DomainCategory(TextDataset):
  def data_path(self):
    raise NotImplementedError, "Deriving class must implement this"

  def ts_byte(self):
    path = self.data_path()
    ts = {}
    for dirpath, dirnames, filenames in os.walk(path):
      for filename in filenames:
        domain = os.path.basename(os.path.dirname(dirpath))
        category = os.path.basename(dirpath)
        with open(os.path.join(dirpath, filename)) as f:
          instance_id = "-".join((domain, category, filename))
          ts[instance_id] = f.read()
    return ts

  def cm_domain(self):
    path = self.data_path()
    cm = {}
    for dirpath, dirnames, filenames in os.walk(path):
      for filename in filenames:
        domain = os.path.basename(os.path.dirname(dirpath))
        category = os.path.basename(dirpath)
        instance_id = "-".join((domain, category, filename))
        cm[instance_id] = [domain]
    return cm
  
  def cm_category(self):
    path = self.data_path()
    cm = {}
    for dirpath, dirnames, filenames in os.walk(path):
      for filename in filenames:
        domain = os.path.basename(os.path.dirname(dirpath))
        category = os.path.basename(dirpath)
        instance_id = "-".join((domain, category, filename))
        cm[instance_id] = [category]
    return cm

  def cm_domaincategory(self):
    path = self.data_path()
    cm = {}
    for dirpath, dirnames, filenames in os.walk(path):
      for filename in filenames:
        domain = os.path.basename(os.path.dirname(dirpath))
        category = os.path.basename(dirpath)
        instance_id = "-".join((domain, category, filename))
        cm[instance_id] = ['-'.join((domain, category))]
    return cm


class ByteUnigram(TextDataset):
  def fm_byte_unigram(self):   return self.features('byte', ext.unigram)

class ByteBigram(TextDataset):
  def fm_byte_bigram(self):    return self.features('byte', ext.bigram)

class ByteTrigram(TextDataset):
  def fm_byte_trigram(self):   return self.features('byte', ext.trigram)

class ByteQuadgram(TextDataset):
  def fm_byte_quadgram(self):  return self.features('byte', ext.quadgram)

class BytePentagram(TextDataset):
  def fm_byte_pentagram(self): return self.features('byte', ext.pentagram)

class ByteUBT(ByteUnigram, ByteBigram, ByteTrigram): pass
