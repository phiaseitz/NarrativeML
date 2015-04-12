# Feature extractors
# Accepts a tokenstream, produces a feature map

def as_strings(seq):
  """
  Convert a sequence of items to a sequence of strings
  Handle 3 cases:
    1) already a sequence of strings
    2) sequence of sequences of strings
    3) other
  """
  for item in seq:
    if isinstance(item, basestring):
      yield item
    try:
      yield '-'.join(item)
    except TypeError:
      yield repr(item)

from hydrat.common.tokenizers import NGram
from hydrat.common.counter import Counter
def ngram_dist(n, ts):
  if isinstance(ts, list):
    # Need to handle lists specially as a slice of a list is a list,
    # which is unhashable and thus incompatible with counter.
    seq = NGram(n)(tuple(ts))
  else:
    seq = NGram(n)(ts)
  return Counter(as_strings(seq))

def unordered_ngram_dist(n, ts):
  return Counter(tuple(sorted(t)) for t in NGram(n)(ts))

def unigram(ts):   return ngram_dist(1, ts) 
def bigram(ts):    return ngram_dist(2, ts) 
def trigram(ts):   return ngram_dist(3, ts)
def quadgram(ts):  return ngram_dist(4, ts)
def pentagram(ts): return ngram_dist(5, ts)
def hexagram(ts): return ngram_dist(6, ts)
def septagram(ts): return ngram_dist(7, ts)
def octagram(ts): return ngram_dist(8, ts)
def nonagram(ts): return ngram_dist(9, ts)
def decagram(ts): return ngram_dist(10, ts)

def unordered_bigram(ts): return unordered_ngram_dist(2, ts)
