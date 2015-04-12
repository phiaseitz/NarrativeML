import os
import numpy
from itertools import izip

def run_command(cmd):
  process = os.popen(cmd)
  output = process.read()
  return_value = process.close()
  if return_value:
    raise ValueError, "Command failed with return %s"%(str(return_value))
  return output

def sparse2sparse_dict(v):
  v.sort_indices()
  return dict((int(k), float(v)) for k,v in izip(v.indices, v.data))

def sparse2dense_dict(v):
  d = dict((a, 0) for a in range(v.shape[1]))
  v.sort_indices()
  for i, val in izip(v.indices, v.data):
    d[i] = val
  return d
