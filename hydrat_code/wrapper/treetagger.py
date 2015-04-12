"""
Dataset subclass for treetagger, providing the ts_treetaggerpos
tokenstream. Uses the treetaggerwrapper of Laurent Pointal
http://perso.limsi.fr/Individu/pointal/doku.php?id=dev:treetaggerwrapper

TODO: Multiprocessing support for parallel TT instances.
"""
import hydrat.external.treetaggerwrapper as ttw
from hydrat import config
from hydrat.dataset.encoded import EncodedTextDataset
from hydrat.configuration import Configurable, DIR
from hydrat.common.pb import ProgressIter
from itertools import izip

import logging
import multiprocessing as mp
import tempfile
import os

logger = logging.getLogger(__name__)

def init_tagger(tt_path, wl_path):
  global __tagger, __wordlist
  with open(wl_path) as f:
    __wordlist = set(w.strip() for w in f)
  __tagger = ttw.TreeTagger(TAGLANG='en', 
        TAGDIR=tt_path, TAGINENC='utf-8', TAGOUTENC='utf-8' )
  logger.debug('initialized TreeTagger instance')

def get_tagger():
  """
  Convenience for testing use
  """
  tt_path = config.getpath('tools','treetagger')
  tagger = ttw.TreeTagger(TAGLANG='en', 
        TAGDIR=tt_path, TAGINENC='utf-8', TAGOUTENC='utf-8' )
  return tagger

def treetagger_pos(item):
  global __tagger
  key, path = item
  with open(path) as f:
    text = f.read()
  try:
    # We replace all ocurrances of '<' with '>' to destroy any accidental SGML
    tokens = [t.split('\t')[1] for t in __tagger.TagText(text.replace('<','>')) if '\t' in t]
  except ttw.TreeTaggerError, e:
    raise ValueError, "Error processing {0}: {1}".format(key, e)
  return key, tokens 

def treetagger_posfw(item):
  """
  Based on the research of Wong et al 2011, we consider not just POS stream,
  but also POS+FW stream. The intuition behind this is that POS tagging is 
  being used to delete topicality, but function words are not topical, so we
  don't want to overgeneralize on them.
  """
  global __tagger, __wordlist
  key, path = item
  with open(path) as f:
    text = f.read()
  try:
    # We replace all ocurrances of '<' with '>' to destroy any accidental SGML
    tokens = []
    for t in __tagger.TagText(text.replace('<','>')):
      if '\t' in t:
        word, pos, lemma = t.split('\t')
        tokens.append(lemma if lemma in __wordlist else pos)
  except ttw.TreeTaggerError, e:
    raise ValueError, "Error processing {0}: {1}".format(key, e)
  return key, tokens 

def treetagger_lemmapos(item):
  """
  Lemma_pos tagstream. Map each token onto lemma+pos, separated
  by underscore
  """
  global __tagger
  key, path = item
  with open(path) as f:
    text = f.read()
  try:
    # We replace all ocurrances of '<' with '>' to destroy any accidental SGML
    tokens = []
    for t in __tagger.TagText(text.replace('<','>')):
      if '\t' in t:
        word, pos, lemma = t.split('\t')
        tokens.append('_'.join((lemma, pos)))
  except ttw.TreeTaggerError, e:
    raise ValueError, "Error processing {0}: {1}".format(key, e)
  return key, tokens 

def write_codepoint(target, outq):
    text = target.ts_codepoint()
    for key in ProgressIter(text,'ToDisk'):
      handle, path = tempfile.mkstemp(dir=config.getpath("paths", "scratch"), prefix="hydrat", suffix="_raw")
      os.write(handle, text[key].encode('utf8'))
      os.close(handle)
      outq.put((key, path))

class TreeTagger(Configurable, EncodedTextDataset):
  requires =\
    { ('tools','treetagger')      : DIR('treetagger')
    }

  def ts_treetaggerpos(self):
    return self.treetagger(treetagger_pos)

  def ts_treetaggerposfw(self):
    return self.treetagger(treetagger_posfw)

  def ts_treetaggerlemmapos(self):
    return self.treetagger(treetagger_lemmapos)

  def treetagger(self, tokenizer):
    # Based on hydrat.dataset.tokenstream.Genia
    # The problem here is that if ts_codepoint is big enough, when it gets
    # read into memory we no longer have enough available to start up the
    # multiprocessing pool.The long term fix is to avoid loading ts_codepoint into memory
    # all at once. A much better interface design would be to have an iterator.
    # For now this requires fixing too many things, so we will have to work
    # around it. 
    # The solution is slightly messy because of the way garbage collection interacts
    # with the underlying OS. even if we gc the ts after writing it to disk, the
    # os doesn't actually get the memory back and so the pool start-up fails.
    # What we have done is run the writing in a separate subprocess, so that
    # we actually get our memory back when it is done. This is a bit messy, but
    # works.
    tt_path = config.getpath('tools','treetagger')
    wl_path = config.getpath('paths','stopwords')
    from multiprocessing import Process, Queue
    num_items = len(self.instance_ids)
    q = Queue(num_items)
    p = Process(target=write_codepoint, args=(self, q))

    p.start()
    items = [ q.get() for i in range(num_items)]
    p.join()

    def text_iter():
      for i in ProgressIter(items,'TreeTagger'):
        yield i

    pool = mp.Pool(config.getint('parameters','job_count'), init_tagger, 
        (tt_path,wl_path))
    vals = pool.imap_unordered(tokenizer, text_iter())
    #init_tagger(tt_path)
    #from itertools import imap
    #vals = imap(treetagger_pos, text_iter())
    streams = dict(vals)

    if config.getboolean('debug', 'clear_temp_files'):
      # Delete all the temporary files
      for k, p in ProgressIter(items,"DeleteTemp"):
        os.unlink(p)
    return streams
      
