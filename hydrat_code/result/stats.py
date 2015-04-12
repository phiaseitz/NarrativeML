import numpy as np
from scipy.stats import chi2
def mcnemar_test(ec_a, ec_b):
  """
  McNemar's test. Accepts multidimensional arrays as well, so we can compute the p-value
  for multiple classes at once.

  see
  Detterich, Thomas G, Statistical Tests for Comparing Supervised Classification Learning Algorithms. ( 1997).

  NOTE: Classes with no instances return a p-value of 0
  """
  nc_a = np.logical_not(ec_a)
  nc_b = np.logical_not(ec_b)

  n00 = np.logical_and(nc_a, nc_b).sum(axis=0)
  n01 = np.logical_and(nc_a, ec_b).sum(axis=0)
  n10 = np.logical_and(ec_a, nc_b).sum(axis=0)
  n11 = np.logical_and(ec_a, ec_b).sum(axis=0)

  # Ignore division by zero on the computation of the statistic, it is the result of unused classes
  prev_set = np.seterr(divide='ignore')
  stat = np.square(np.abs(n01 - n10) - 1 ) / np.array(n01+n10, dtype=float)
  np.seterr(**prev_set)

  rv = chi2(1)
  p = rv.sf(stat)
  return p

def mcnemar(interpreter, tsr_a, tsr_b, perclass = False):
  if perclass:
    correct_a = np.nansum(tsr_a.overall_correct(interpreter),axis=2)
    correct_b = np.nansum(tsr_b.overall_correct(interpreter),axis=2)
    return mcnemar_test(correct_a, correct_b)
  else:
    ec_a = np.nansum(tsr_a.overall_correct(interpreter),axis=2).all(axis=1).astype(bool)
    ec_b = np.nansum(tsr_b.overall_correct(interpreter),axis=2).all(axis=1).astype(bool)
    return mcnemar_test(ec_a, ec_b)
    
# Added by Li Wang (li@liwang.info)
def randomisation(interpreter, tsr_a, tsr_b, N = 10000):
  from hydrat import rng
  ''' Randomisation test or Randomised Estimation based on:
  Alexander Yeh. 2000. More accurate tests for the statistical significance of result differences. In Proceedings of the 18th International Conference on Computational Linguistics (COLING 2000), pages 947--953, Saarbrucken, Germany.'''
  ec_a = np.nansum(tsr_a.overall_correct(interpreter),axis=2).all(axis=1).astype(bool)
  ec_b = np.nansum(tsr_b.overall_correct(interpreter),axis=2).all(axis=1).astype(bool)
  assert len(ec_a) ==  len(ec_b)
  acc_a = float(ec_a.sum())/float(len(ec_a))
  acc_b = float(ec_b.sum())/float(len(ec_b))
  assert acc_a != acc_b
  ec_better, ec_worse = (ec_a, ec_b) if acc_a>acc_b else (ec_b, ec_a)
  acc_better = acc_a if acc_a>acc_b else acc_b
  n = 0
  for i in xrange(N):
    ecbetter = []
    ecworse = []
    for j, v in enumerate(ec_better):
      if rng.rand() >= 0.5:
        ecbetter.append(v)
        ecworse.append(ec_worse[j])
      else:
        ecbetter.append(ec_worse[j])
        ecworse.append(v)
    acc_random = float(sum(ecbetter))/float(len(ecbetter)) 
    if acc_random >= acc_better:
      n += 1
  return float(n+1)/float(N+1)
