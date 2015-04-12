"""
Discretization functions
"""

import numpy

def bernoulli(v):
  nonzero = numpy.zeros(v.shape, dtype=bool)
  nonzero[v.nonzero()] = True
  zero = numpy.logical_not(nonzero)
  retval = numpy.concatenate((zero[None], nonzero[None]))
  return retval

class UniformBand(object):
  def __init__(self, bands):
    raise NotImplementedError, "need to update"
    self.__name__ = 'uniform%dband' % bands
    self.bands = bands

  def __call__(self, v):
    limit = float(numpy.max(v.data) + 1)
    bins = numpy.digitize(v, numpy.arange(0, limit, limit/self.bands))
    r = numpy.empty((self.bands, len(v)), dtype=bool)
    for i in range(self.bands):
      r[i] = (bins == (i+1))
    return r

class EquisizeBand(object):
  def __init__(self, bands):
    raise NotImplementedError, "need to update"
    self.__name__ = 'equisize%dband' % bands
    self.bands = bands

  def __call__(self, v):
    r = numpy.empty((self.bands, v.shape[0]), dtype=bool)
    band_size = 100.0 / (self.bands)
    for i in range(self.bands):
      r[i] = numpy.logical_and( (i * band_size) <= v, v < (i * (band_size + 1)) )
    return r
