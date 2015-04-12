# Very loosely based on http://www.mit.edu/~mokelly/devlog/bootstrap.py
# and http://www-i6.informatik.rwth-aachen.de/~bisani/bootlog.html
import random
import time
import numpy

def resampleArray(data):
  d = numpy.array(data, copy = 0)
  sample_size = len(d)
  # Choose #sample_size members of d at random, with replacement
  choices = numpy.random.random_integers(0, sample_size-1, sample_size)
  sample = d[choices]
  return sample 

# The format for the PMF is a dict-like object mapping from a label to the
# number of instances of that label. Returns a sample with the same labels
# and the number of instances of that label in that sample
def resamplePMF(data):
  total_instances = sum(data.values())
  labels = []
  CDF    = numpy.empty(len(data)) 
  curr_thresh = 0.0
  for i,label in enumerate(data):
    curr_thresh += float(data[label]) / float(total_instances)
    labels.append(label)
    CDF[i] = curr_thresh
  
  sample = dict.fromkeys(labels, 0)
  for r in numpy.random.random(total_instances):
    for i, thresh in enumerate(CDF):
      if r < thresh:
        sample[labels[i]] += 1
        break
  assert(sum(sample.values()) == total_instances)
  return sample

def compartmentResample(data):
  sample = {}
  for compartment in data:
    compartment_sample = resamplePMF(compartment)
    for key in compartment_sample:
      sample[key] = sample.get(key, 0) + compartment_sample[key]
  return sample

class Bootstrap(object):
  def __init__(self, resampler, function, seed=None):
    if seed:
      self.seed = seed
    else:
      self.seed = time.time()
    numpy.random.seed(seed)
    self.resampler = resampler
    self.function = function
  
  def bootstrap_t(self, data, alpha=0.95, B=2000):
    data_points = numpy.array( [   self.function(self.resampler(data))
                               for i
                               in  xrange(B)
                               ]
                             )
    quantiles = numpy.argsort(data_points)
    i = int(round(alpha * B / 2))
    lower = data_points[quantiles[  i]]
    upper = data_points[quantiles[B-i]]
    return (lower, upper) 


if __name__ == "__main__":
  resampler = Bootstrap(resampleArray, sum)
  print resampler.bootstrap_t([1,2,3,4,5], B=20000)
  resampler = Bootstrap(resamplePMF, lambda x: sum(key*x[key] for key in x))
  print resampler.bootstrap_t({1:1,2:1,3:1,4:1,5:1}, B=20000)
  
