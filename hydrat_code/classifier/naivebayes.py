"""
Mutlinomial Naive Bayes
see McCallum and Nigam (1998) "A Comparison of Event Models for Naive Bayes Text Classification"

..todo: Implement other event models

Marco Lui Feb 2011
"""
import numpy as np
import scipy.sparse as sp

from hydrat.classifier.abstract import Learner, Classifier

# TODO: memoize logfac
def logfac(a):
  return np.sum(np.log(np.arange(1,a+1)))
logfac = np.frompyfunc(logfac, 1, 1)

class NBEventModel(object):
  """
  Abstract algorithm class for Naive Bayes event models.
  Implements two methods, for calculaing class priors and 
  term priors on the basis of the feature and class matrixes.
  Each method should return the log(P(C)) and log(P(t|C)) 
  respectively.
  Assumes that both the feature and the class matrix are
  scipy.sparse.csr_matrix instances.
  """
  def __init__(self):
    if not hasattr(self, '__name__'):
      self.__name__ = self.__class__.__name__

  def class_priors(self, fm, cm):
    raise NotImplementedError

  def term_priors(self, fm, cm):
    raise NotImplementedError

class Multinomial(NBEventModel):
  def class_priors(self, fm, cm):
    return np.log(cm.sum(0))

  def term_priors(self, fm, cm):
    v = fm.shape[1]
    prod = (fm.T * cm).todense()
    ptc = np.log(1 + prod) - np.log(v + prod.sum(0))
    return ptc
 
class MultinomialBoolean(Multinomial):
  """
  Convert term matrix to boolean before applying multinomial model.
  ..todo: references
  """
  def term_priors(self, fm, cm):
    fm_ = fm.copy()
    fm_.data = (fm_.data > 0).astype('int64')
    return Multinomial.term_priors(self, fm_, cm)

class NaiveBayesL(Learner):
  __name__ = "naivebayes" 
  VALID_CLASS_PRIORS = ['default', 'uniform']
  def __init__(self, eventmodel, class_prior='default', output_prob=False):
    """
    class_prior: override the calculated class prior with a uniform prior
    output_prob: output a probability distribution over classes, rather than just the most
                 likely class
    """

    Learner.__init__(self)
    self.eventmodel = eventmodel
    if class_prior not in self.VALID_CLASS_PRIORS:
      raise ValueError, "unknown class prior"
    self.class_prior = class_prior
    self.output_prob = output_prob

  def is_pickleable(self):
    return True

  def __getstate__(self):
    return (self.eventmodel, self.class_prior, self.output_prob)

  def __setstate__(self, state):
    Learner.__init__(self)
    self.eventmodel, self.class_prior, self.output_prob = state

  def _params(self):
    return {'eventmodel':self.eventmodel.__name__, 'class_prior':self.class_prior, 'output_prob':self.output_prob }

  def _learn(self, fm, cm):
    tot_cl = cm.shape[1]
    used_cl = np.flatnonzero(cm.sum(0) > 0)
    cm = sp.csr_matrix(cm[:,used_cl], dtype='int32')
    if self.class_prior == 'default':
      pc = self.eventmodel.class_priors(fm, cm)
    elif self.class_prior == 'uniform':
      pc = np.log(np.ones((cm.shape[1],),dtype=cm.dtype))
    else:
      raise ValueError, "unknown class prior!"
    ptc = self.eventmodel.term_priors(fm, cm)
    return NaiveBayesC(pc, ptc, tot_cl, used_cl, self.output_prob)

  def _check_installed(self):
    pass

class NaiveBayesC(Classifier):
  __name__ = "naivebayes" 
  def __init__(self, pc, ptc, tot_cl, used_cl, output_prob):
    """
    pc: class log priors (numpy matrix (1,num_class))
    ptc: term log priors (numpy matrix ax0:term ax1:class)
    tot_cl: total number of classes in the larger class space
    used_cl: indexes of classes used in the larger class space
    """
    Classifier.__init__(self)
    self.pc = pc
    self.ptc = ptc
    self.tot_cl = tot_cl
    self.used_cl = used_cl
    self.output_prob = output_prob

  def is_pickleable(self):
    return True

  # Implementation of getstate/setstate based on liblinear.py
  def __getstate__(self):
    initargs = (self.pc, self.ptc, self.tot_cl, self.used_cl, self.output_prob)
    metadata = self.metadata
    return (initargs, metadata)

  def __setstate__(self, state):
    initargs, metadata = state
    self.__init__(*initargs)
    self.metadata.update(metadata)
    

  def _classify(self, fm):
    pdc = fm * self.ptc
    pd = pdc + self.pc
    if self.output_prob:
      pd = np.array(pd)
      with np.errstate(under='ignore'):
        cm = (1/np.exp(pd[:,:,None] - pd[:,None,:]).sum(1))
    else:
      cl = np.argmax(pdc + self.pc, axis=1)
      cm = np.zeros((fm.shape[0], self.tot_cl), dtype=bool)
      cm[np.arange(cm.shape[0]), self.used_cl[np.array(cl)[:,0]]] = True
    return cm

def multinomialL(*args, **kwargs): return NaiveBayesL(Multinomial(), *args, **kwargs)
