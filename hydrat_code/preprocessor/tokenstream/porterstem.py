import re

from hydrat.common.PorterStemmer import PorterStemmer

RE_WORD = re.compile(r'(?P<word>\w+)(\W|\Z)')
class PorterStemTagger(PorterStemmer):
  def __init__(self, re_word = RE_WORD):
    PorterStemmer.__init__(self)
    self.re_word = re_word

  def process(self, text):
    tokens = []
    pos = 0
    match = self.re_word.search(text[pos:])
    while match:
      start = match.start('word') + pos
      end = match.end('word') + pos
      token = dict()
      word = match.group('word')
      token['word'] = word
      token['start'] = start
      token['end'] = end
      token['stem'] = self.stem(word, 0, len(word)-1)
      tokens.append(token)
      pos = end
      match = self.re_word.search(text[pos:])
    return tokens
    
