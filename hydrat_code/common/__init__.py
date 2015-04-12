import numpy
from time import time

def proportion_allocate(itemlist, proportions, randomise = False, randomseed = None):
  try: 
    denominator = float(sum(proportions))
  except TypeError:
    raise TypeError, "Proportions not number values"
  totalitems = len(itemlist)
  partitionsizes = map(lambda x: int(x * totalitems // denominator) , proportions)
  partitionsizes[0] = totalitems - sum(partitionsizes[1:]) #Account for rounding errors by favoring first partition
  partitions = []
  if randomise:
    import random
    if randomseed != None:
      random.seed(randomseed)
    itemlist = itemlist[:]
    random.shuffle(itemlist)
  for i in xrange(len(partitionsizes)):
    partition_items = itemlist[:partitionsizes[i]]
    itemlist = itemlist[partitionsizes[i]:]
    partitions.append(partition_items)
  assert len(itemlist) == 0, "Elements left in itemlist: " + str(len(itemlist))
  return partitions

def progress(seq, count, callback):
  assert callable(callback)
  assert isinstance(count, int)
  call_indices = []
  step = len(seq) / float(count)
  curr = 0 
  while curr < len(seq):
    call_indices.append(int(curr))
    curr += step
  for i,x in enumerate(seq):
    if i in call_indices: callback(i, len(seq))
    yield x

def timed_report(seq, interval, callback):
  """
  Call a callback if a certain amount of time has elapsed in processing a sequence
  """
  assert callable(callback)
  try:
    total = len(seq)
  except TypeError:
    total = None
  next_call = time() + interval

  for i,x in enumerate(seq):
    t = time()
    if t > next_call:
      callback(i, total)
      next_call = t + interval 
    yield x
    
def entropy(v, axis=0):
  """
  Optimized implementation of entropy. This version is faster than that in 
  scipy.stats.distributions, particularly over long vectors.
  """
  v = numpy.array(v, dtype='float')
  s = numpy.sum(v, axis=axis)
  with numpy.errstate(divide='ignore', invalid='ignore'):
    r = numpy.log(s) - numpy.nansum(v * numpy.log(v), axis=axis) / s
  return r

def as_set(s):
  """Return the argument as a set
  Useful for handling arguments that can be items or sequences of items
  if none, return an empty set
  if the argument is a string or unicode, return a single-element set
  if the item is iterable, return a set
  else return a single-element set
  """
  if s is None: return set()
  elif isinstance(s, str): return set([s])
  elif isinstance(s, unicode): return set([s])
  else: 
    try:
      return set(s)
    except TypeError:
      return set([s])

def rankdata(vector, reverse=False):
  """
  Compute rank order statistics for a 1-d vector.
  
  TODO: proper tie handling
  TODO: generalize to an n-d vector
  """
  result = numpy.empty(len(vector), dtype=int)
  if reverse:
    result[numpy.argsort(vector)] = numpy.arange(len(vector)-1,-1,-1)
  else:
    result[numpy.argsort(vector)] = numpy.arange(len(vector))
  return result


