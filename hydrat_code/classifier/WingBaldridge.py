"""
Implementation of the geolocation method of Wing and Baldridge as a
hydrat classifier. The basic design is a nearest-prototype classifier
using a KL-divergence metric on a word unigram space. The only
slight cleverness is in the use of a pseudo Good-Turing smoothing
approach to deal with words that have 0 count. Here we assume that
the feature space used is a count space - using this classifier on
a word unigram space will most closely approximate Wing and Baldridge 2011
and Roller et al 2012.
"""

import hydrat.common.prototype as prot
from hydrat.classifier.abstract import Learner, Classifier
import numpy as np
import scipy.sparse as sp

def kl_divergence(v1, v2):
  _p = np.array(v1)
  _q = np.array(v2)
  used = (_p.sum(0) != 0)
  p = _p[:,used]
  q = _q[:,used]

  #x = ((p*np.log(p))[:,None] - p[:,None]*np.log(q)).sum(2)
  # We end up unrolling this for memory reasons
  retval = np.empty((p.shape[0], q.shape[0]))
  plp = (p*np.log(p)).sum(1)
  for i,q_ in enumerate(q):
    retval[:,i] = plp - (p*np.log(q_)).sum(1)
  return retval

def global_dist(fm):
  """
  Compute the global distribution given a feature map,
  assuming axis 0 is instances and axis 1 is features.
  Basically sum and normalize - maybe there is an inbuilt for this?
  """
  return np.array(fm.sum(axis=0))[0] / float(fm.sum())

def good_turing_smoothing(fm, theta_d=None):
  """
  Good-Turing smoothing as described in Wing & Baldridge 2011.
  fm is a scipy.sparse.csr_matrix with instances on axis0
  and features on axis1. Each value is a feature count for
  an instance. The basic idea is that the word distribution
  of a document backs off to the global distribution.

  Note that the smoothing ensures that all parameters have
  a nonzero estimate, so we return a dense matrix

  theta_d is the back-off prior. We leave it as a parameter
  as we may or may not want to learn it from the training data.

  What is not mentioned in Wing & Baldridge 2011 but can be 
  discovered from the provided source code 
  (http://textgrounder.googlecode.com/hg/python/wikigrounder/word_distribution.py)
  is that this version of Good-Turing smoothing is troublesome
  in particular scenarios: 
  (1) very many terms seen only once - resulting in too much mass being
      redistributed to 0-terms
  (2) no terms seen only once - resulting in no mass to redistribute
  (3) no unseen terms - resulting in no terms to redistribute reserved
      mass to

  To address (1), Wing arbitrarily limits alpha_dk to 0.5. 
  To address (2), Wing sets the numerator for alpha_dk to 1 if it is zero
  It's not clear how (3) is addressed, but we handle it by not redistributing
  any mass in that case.
  """
  # document size
  d_sum = np.array(fm.sum(axis=1))[:,0]

  # global distribution
  if theta_d is None:
    theta_d = global_dist(fm)
  elif theta_d.shape[0] != fm.shape[1]:
    raise ValueError("length of theta_d does not match number of features")

  # per-document distribution
  theta_di = fm.asfptype().tocsc()
  theta_di.data /= d_sum[theta_di.indices]

  # Precomputed normalization factor
  # We are simply working out the total global mass accounted for 
  # by the terms not present in each document. This is computed
  # as 1- the mass for the terms present, as the set of terms 
  # present is expected to be much smaller than the vocabulary
  u_di = 1 - np.array(sp.csr_matrix((theta_d[fm.indices],fm.indices.copy(), fm.indptr.copy())).sum(axis=1))[:,0]

  # alpha_dk is the probability mass reserved for unseen words.
  # it is the proportion of tokens seen only once in a document 
  # to the total number of tokens in a document
  one_c = np.array(sp.csr_matrix((fm.data==1,fm.indices.copy(),fm.indptr.copy())).sum(axis=1), dtype=float)[:,0]
  # Like Wing's implementation, we limit the amount of mass to redistribute to 0.5
  # if no terms are seen only once, we set the numerator to 1. Where u_di is 0,
  # we have no features to redistribute mass to, and so we set alpha_dk to zero.
  # It is possible in rare cases for d_sum = 0 . In this case, we manually
  # set alpha_dk to 1 to completely back off to the prior.
  with np.errstate(divide='ignore'):
    alpha_dk = np.minimum(np.maximum(one_c,1) / d_sum, 0.5) * u_di.astype(bool)
    alpha_dk[d_sum == 0] = 1

  # Compute how to distribute mass for 0 counts 
  # it is possible for u_di to be 0, if a document(or pseudo-document) contains
  # every term in the vocabulary. In this case, no mass from that document needs
  # to be distributed.
  with np.errstate(invalid='ignore', divide='ignore'):
    z_mass = np.logical_not(fm.toarray()) * theta_d * (alpha_dk / u_di)[:,None]

  # NaNs result in the rows where there is no 0-mass to redistribute. We set this
  # back to 0.
  z_mass[np.isnan(z_mass)] = 0

  # Smoothed value for nonzero-counts
  nz_mass = theta_di.copy()
  nz_mass.data *= (1-alpha_dk)[nz_mass.indices]

  return z_mass + nz_mass

class WingBaldridgeL(Learner):
  PRIORS = ['train', 'test', 'both']
  __name__ = "WingBaldridge2011"
  def __init__(self, global_prior='train'):
    if global_prior not in self.PRIORS:
      raise ValueError("unknown global prior %s".format(global_prior))
    self.global_prior = global_prior
    Learner.__init__(self)

  def __getstate__(self):
    return self._params()

  def __setstate__(self, state):
    self.__init__(**state)

  def _params(self):
    return {'global_prior':self.global_prior}

  def _learn(self, fm, cm):
    prototype = prot.sum()
    self.logger.debug("computing class prototypes")
    pf, pc = prototype.class_prototypes(fm, cm)
    self.logger.debug("computing global dist")
    theta_d = global_dist(fm)
    self.logger.debug("returning classifier")
    return WingBaldridgeC(pf, pc, theta_d, self.global_prior)

  def _check_installed(self):
    pass

class WingBaldridgeC(Classifier):
  __name__ = "WingBaldridge2011"

  def __init__(self, pf, pc, theta_d, global_prior):
    self.pf = pf
    self.pc = pc
    self.theta_d = theta_d
    self.global_prior = global_prior
    Classifier.__init__(self)

  def _classify(self, fm):
    self.logger.debug("computing prior")
    # First step is to decide on the global prior to use
    if self.global_prior == 'train':
      theta_d = self.theta_d
    elif self.global_prior == 'test':
      theta_d = global_dist(fm)
    elif self.global_prior == 'both':
      # for some reason it is not possible to vstack csr_matrix instances,
      # so we do the summation manually
      theta_d = global_dist(self.pf.sum(0)+fm.sum(0))
    else:
      raise ValueError("unknown global prior %s".format(self.global_prior))

    # Based on the global prior, we trim the feature map
    feats = np.flatnonzero(theta_d)
    theta_d = theta_d[feats]

    # Next we perform the smoothing - over the test documents, as well as 
    # the train pseudodocuments (the class prototypes)
    self.logger.debug("smoothing prototypes")
    pf = good_turing_smoothing(self.pf[:,feats], theta_d)
    self.logger.debug("smoothing test instances")
    fm = good_turing_smoothing(fm[:,feats], theta_d)

    self.logger.debug("computing KL divergence")
    distance_matrix = kl_divergence(pf, fm)
    self.logger.debug("assigning classes")
    nn = distance_matrix.argmin(axis=0)
    retval = self.pc[nn]
    return retval

if __name__ == "__main__":
  print "Testing WingBaldridge implementation"
  fm = sp.csr_matrix([
    [ 2, 1, 1, 0, 1],
    [ 1, 2, 1, 0, 1],
    [ 1, 1, 1, 1, 0],
    [ 4, 3, 0, 0, 4],
    [ 1, 2, 0, 1, 4],
    [ 2, 2, 1, 0, 0],
  ])
  cm = np.array([
    [0,1],
    [0,1],
    [1,0],
    [1,0],
    [1,0],
    [0,1],
  ], dtype=bool)

  l = WingBaldridgeL()
  c = l(fm, cm)
  print c(fm)
  import pdb;pdb.set_trace()


