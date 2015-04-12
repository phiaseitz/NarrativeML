from hydrat.dataset.encoded import EncodedTextDataset
from hydrat.preprocessor.tokenstream.porterstem import PorterStemTagger
from hydrat.common.pb import ProgressIter
from hydrat.wrapper.genia import GeniaTagger
from hydrat import config
from hydrat.configuration import Configurable, EXE, DIR

import multiprocessing as mp
from itertools import izip
import logging

logger = logging.getLogger(__name__)

#TODO: Keep partial tokenstreams in shelve objects in scratch space 
#      - this will avoid recomputation if it cuts out halfway somehow.
 
class PorterStem(EncodedTextDataset):
  def ts_porterstemmer(self):
    text = self.ts_codepoint()
    stemmer = PorterStemTagger()
    streams = dict( (i,stemmer.process(text[i].encode('utf8'))) for i in ProgressIter(text,'Porter Stemmer') )
    return streams

def init_tagger(tagger_exe, genia_path):
  global __tagger
  __tagger = GeniaTagger(tagger_exe, genia_path)
  logger.debug('initialized GeniaTagger instance')
  
def genia_full(text):
  global __tagger
  tokens = __tagger.process(text)
  return tokens 

def genia_pos(text):
  global __tagger
  tokens = __tagger.process(text, find_span=False)
  return [ t.POStag for t in tokens ] 

class Genia(Configurable, EncodedTextDataset):
  requires =\
    { ('tools','genia')      : EXE('geniatagger')
    , ('tools','genia_data') : DIR('geniatagger-3.0.1')
    }

  def genia_tag(self, tag_fn):
    text = self.ts_codepoint()
    keys = text.keys()
    def text_iter():
      for i in ProgressIter(keys,'GENIA Tagger'):
        yield text[i].encode('utf8')

    tagger_exe = config.getpath('tools','genia')
    genia_path = config.getpath('tools','genia_data')
    pool = mp.Pool(config.getint('parameters','job_count'), init_tagger, 
        (tagger_exe, genia_path))
    vals = pool.imap(tag_fn, text_iter())
    streams = dict(izip(keys, vals))
    return streams

  def ts_genia(self):
    return self.genia_tag(genia_full)

  def ts_geniapos(self):
    return self.genia_tag(genia_pos)

