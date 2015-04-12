"""
Implementation of a disk-backed dictionary module.
This module is not intended to be used for persistence; its job
is to provide a mapping interface to a collection of entries
that are too large to fit in memory, but whose lifespan
is inferior to that of the program.

Note that __getitem__ returns a copy of the item rather than
a reference, so any modifications made to the item must then 
be explicitly committed by setting the item again. This behaviour
differs from that of a standard dictionary.
"""

import tempfile
import collections
import shutil
import os
import marshal

class diskdict(collections.MutableMapping):
  def __init__(self, temp_path=None):
    self.directory = tempfile.mkdtemp(dir=temp_path, suffix='-diskdict')
    self.key_map = {}

  def __del__(self):
    if hasattr(self, 'directory'):
      shutil.rmtree(self.directory)

  def __iter__(self):
    return self.key_map.iterkeys()
  
  def __len__(self):
    return len(self.key_map)

  def __getitem__(self, key):
    return marshal.load(open(self.key_map[key]))
    
  def __setitem__(self, key, value):
    if key in self.key_map:
      del self[key]
    handle, path = tempfile.mkstemp(dir=self.directory)
    marshal.dump(value, os.fdopen(handle, 'w')) 
    self.key_map[key] = path

  def __delitem__(self, key):
    # delete existing value
    os.remove(self.key_map[key])
    del self.key_map[key]

  def keys(self):
    return self.key_map.keys()

if __name__ == "__main__":
  x = diskdict()
  x['test'] = {'a':1,'b':2,'c':3}
  x['exam'] = {'a':4,'b':5,'c':6}
  print x['test']
  print x['exam']
  x['test'] = {'a':7,'b':8,'c':9}
  print x['test']
