import numpy

from result import Result
from result import confusion_matrix, classification_matrix

class ResultError(Exception): pass


class Microaverage(object):
  def __call__(self, matrix, function = None):
    micro_matrix = matrix.sum(axis=0)
    return micro_matrix if function is None else function(micro_matrix)

class Macroaverage(object):
  def __call__(self, matrix, function = None):
    self.ignored_classes = [] 
    if function is None:
      return matrix.mean(axis=0)
    else:
      _result = []
      for index, row in enumerate(matrix):
        row_result = function(row)
        #print function, row, '-->', row_result
        if numpy.isnan(row_result).any():
          #Undefined value, means we should ignore this class
          self.ignored_classes.append(index) 
          pass
        else:
          _result.append(row_result)
          
      result = numpy.array(_result).mean(axis=0)
      return result

class ConfusionMatrixMetric(object):
  def parse(self, matrix):
    assert matrix.shape == (4,)
    self.tp, self.tn, self.fp, self.fn = matrix

  def __call__(self, matrix):
    self.parse(matrix)
    return self.compute()

  def compute(self):
    raise NotImplementedError

class Precision(ConfusionMatrixMetric):
  def compute(self):
    # Undefined if there are no goldstandard positives
    if self.tp == self.fn == 0:
      return numpy.nan 
    elif self.tp == self.fp == 0:
      return 0
    else:
      return self.tp / float(self.tp + self.fp)
    
class Recall(ConfusionMatrixMetric):
  def compute(self):
    # Undefined if there are no goldstandard positives
    if self.tp == self.fn == 0:
      return numpy.nan 
    else:
      return self.tp / float(self.tp + self.fn)

def fscore(p, r, b):
  try:
    return (1 + b ** 2) * p * r / float( b ** 2 * p + r ) 
  except ZeroDivisionError:
    return 0

class FScore(Precision, Recall):
  def __init__(self, beta= 1.0):
    self.beta = beta

  def compute(self):
    # Undefined if there are no goldstandard positives
    if self.tp == self.fn == 0:
      return numpy.nan 
    # Zero if there are no True Positives
    if self.tp == 0:
      return 0.0 
    p = Precision.compute(self)
    r = Recall.compute(self)
    f = fscore(p, r, self.beta)
    return f

class PRF(FScore):
  def compute(self):
    # Undefined if there are no goldstandard positives
#    if self.tp == self.fn == 0:
#      return numpy.nan 
    # Zero if there are no True Positives
    if self.tp == 0:
      return numpy.array((0.0, 0.0, 0.0))
    p = Precision.compute(self)
    r = Recall.compute(self)
    f = fscore(p,r,self.beta)
    return numpy.array((p,r,f))

class MCC(ConfusionMatrixMetric):
  """
  Matthew's Correlation Coefficient. Described in (1) and used for the BioCreative series of shared tasks (2).

  1. Matthews B. Comparison of the predicted and observed secondary structure of T4 phage lysozyme. 
     Biochimica et Biophysica Acta (BBA) - Protein Structure. 1975;405(2):442-451. 
     Available at: http://dx.doi.org/10.1016/0005-2795(75)90109-9.
  2. Leitner F, Mardis Sa, Krallinger M, et al. An Overview of BioCreative II.5. 
     IEEE/ACM transactions on computational biology and bioinformatics / IEEE, ACM. 2010;7(3):385-99. 
     Available at: http://www.ncbi.nlm.nih.gov/pubmed/20704011.
  """
  def compute(self):
    tp, tn, fp, fn = map(int,(self.tp, self.tn, self.fp, self.fn))
    x = tp*tn-fp*fn
    y = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return x / numpy.sqrt(y) 

class CombinedMicroAverage(object):
  def __init__(self, matrix):
    n_res, n_class, n_items = matrix.shape
    assert n_items == 4 #sanity check, this axis is tp, tn, fp, fn
    self.combined_matrix = matrix.sum(axis = 0)
    self.aggregator = Microaverage()

  def __iter__(self):
    """
    K-V pair iterator allows us to consume this in the dict constructor
    """
    return iter([('precision',self.precision),('recall',self.recall),('fscore',self.fscore)])
    
  def __repr__(self):
    return "<P:%.3f R:%.3f F:%.3f>"% (self.precision, self.recall, self.fscore)

  @property
  def precision(self):
    return self.aggregator(self.combined_matrix, Precision())

  @property
  def recall(self):
    return self.aggregator(self.combined_matrix, Recall())

  @property
  def fscore(self):
    return self.aggregator(self.combined_matrix, FScore())

class CombinedMacroAverage(object):
  """
  tbaldwin:  Well, putting aside the issue of bootstraps etc for now, 
             and trying to come up with a single p/r/f 3-tuple, we can 
             first collapse the TP/FP/FN/TN counts across the 10 folds
  tbaldwin:  Simple sum at this point
  tbaldwin:  If we are macro-averaging, we then calculate the p and r 
             for each class, and calculate the simple arithmetic mean 
             of each to get the macro-averaged (overall) p and r
             From this, we calculate the (overall) f directly using 
             the standard 2pr/(p+r) formula
             tbaldwin:  No -- we're never microaveraging
  me:        the summation across the 10 folds is basically a microaverage except that we're not scaling by 10 no?
             an alternative approach would be to compute a macroaverage on each fold and then report the average of that? is it equivalent?
  tbaldwin:  In practice, f should be the macro-average of the individual fs across the different classes, but this can 
             potentially lead to the f not having the expected 2pr/(p+r) vale relative  to the macro-averaged p and r, so we fudge
             Not necessarily, e.g. if you don't get a perfect split across the 10 folds
  me:        hmm so the size of the test set in each fold is slightly different
  tbaldwin:  I see what you're saying about the microaverage
             Yes, you're right
             Not that it's a matter for concern -- what we're doing is standard practice from cross-validation, independent of micro- 
             and macro-averaging
  me:        ok- so summary is: sum the 10 folds, then apply a standard macroaverage to calculate p/r/f
  tbaldwin:  Yup
             Is that what ends up in the code?
  me:        i guess so :) do you ever envision wanting to calculate a different value?
             i'm basically pulling the calculation out into a separate module, so that sets of results report just a 3d confusion 
             matrix
  tbaldwin:  I envision wanting to perform statistical tests over the different folds, yes (a la what you were doing way 
             back, as part of your summer internship)
             And bootstrap needs to be dusted off at some point
  me:        one last question, do you have a name for this p/r/f?
             summed macroaverage?
  tbaldwin:  To distinguish it from the macro-average for each fold you mean?
  me:        yes
  tbaldwin:  How about the combined macro-average?
  me:        very well, here comes class CombinedMacroAverage(object):
  """
  def __init__(self, matrix):
    n_res, n_class, n_items = matrix.shape
    assert n_items == 4 #sanity check, this axis is tp, tn, fp, fn
    self.combined_matrix = matrix.sum(axis = 0)
    self.macroaverage = Macroaverage()

  def __repr__(self):
    return "<P:%.3f R:%.3f F:%.3f>"% (self.precision, self.recall, self.fscore)

  @property
  def precision(self):
    return self.macroaverage(self.combined_matrix, Precision())

  @property
  def recall(self):
    return self.macroaverage(self.combined_matrix, Recall())

  @property
  def fscore(self):
    return self.macroaverage(self.combined_matrix, FScore())
