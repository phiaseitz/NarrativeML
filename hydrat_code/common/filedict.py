"""
FileDict class to provide a mapping interface to
accessing a set of files by key. The implementation
uses __setitem__ to associate a key with a path, and
__getitem__ opens the path and returns the contents
as a string.

Marco Lui
January 2013
"""
from collections import MutableMapping

class FileDict(MutableMapping):
  def __init__(self, *args, **kwargs):
    self.paths = {}
    self.update(*args, **kwargs)

  def update(self, *args, **kwargs):
    for k, v in dict(*args, **kwargs).iteritems():
      self[k] = v

  def __len__(self):
    return len(self.paths)

  def __iter__(self):
    return iter(self.paths)

  def __getitem__(self, key):
    try:
      with open(self.paths[key]) as f:
        return f.read()
    except KeyError:
      raise KeyError(key)

  def __delitem__(self, key):
    del self.paths[key]

  def __setitem__(self, key, value):
    self.paths[key] = value
