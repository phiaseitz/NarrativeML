from hydrat.dataset.text import TextDataset
from hydrat.common.tokenizers import bag_of_words
import hydrat.common.extractors as ext
from hydrat.common.pb import ProgressIter


class WhitespaceWords(TextDataset):
  def ts_word(self):
    text = self.tokenstream('byte')
    streams = dict( (i,bag_of_words(text[i])) for i in ProgressIter(text,'Whitespace Words') )
    return streams

class BagOfWords(WhitespaceWords):
  def fm_word_unigram(self): return self.features('word', ext.unigram)

import nltk.tokenize as tk
def nltkword(text):
  # built-in assumption that words do not exceed 100 characters in length
  return tuple(t for t in tk.word_tokenize(text) if len(t) <= 100)

class NLTKWord(TextDataset):
  def ts_nltkword(self):
    return self.from_byte(nltkword)

  def fm_nltkword_unigram(self):
    return self.features('nltkword', ext.unigram)
