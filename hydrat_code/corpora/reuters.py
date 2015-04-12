"""
http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz
"""
import os
import logging
from time import time
from sgmllib import SGMLParser
from collections import defaultdict

from hydrat import config
from hydrat.dataset.encoded import ASCII
from hydrat.dataset.words import BagOfWords
from hydrat.configuration import Configurable, DIR

logger = logging.getLogger(__name__)

class ReutersParser(Configurable, SGMLParser):
  requires=\
    { ('corpora', 'reuters') : DIR('reut21578')
    }
  def __init__(self, verbose=0):
    SGMLParser.__init__(self,verbose)
    self.reset()

  def reset(self):
    SGMLParser.reset(self)
    self.docmap = {}
    self.classmap = {}
    self.inReutersBlock = False
    self.inTopic = False
    self.addToTopics = False
    self.keepContent = False
    self.content = ""
    self.lewissplit = {"TRAIN":[], "TEST":[], "NOT-USED":[]}

  def start_reuters(self,attrs):
    attrd = dict(attrs)
    self.inReutersBlock = True
    self.currentid = attrd["newid"]
    self.classmap[self.currentid] = []
    self.lewissplit[attrd["lewissplit"]].append(self.currentid)

  def start_topics(self, attrs):
    if self.inReutersBlock:
      self.inTopic = True

  def end_topics(self):
    self.inTopic = False

  def start_d(self, attrs):
    if self.inTopic:
      self.addToTopics = True

  def end_d(self):
    if self.inTopic:
      self.addToTopics = False

  def start_text(self,attrs):
    if self.inReutersBlock:
      self.keepContent = True

  def end_text(self):
    self.keepContent = False
 
  def handle_data(self, data):
    if self.keepContent:
      self.content += data
    if self.addToTopics:
      self.classmap[self.currentid].append(data)

  def end_reuters(self):
    if self.inReutersBlock:
      self.docmap[self.currentid] = self.content
    self.content = ""
    self.currentid = None
    self.inReutersBlock = False
    self.keepContent = False

  def run(self):
    self.reset()
    logger.debug("Parsing reuters data set")
    start_time = time()
    for i in range(0,22):
      with open(os.path.join(config.getpath('corpora', 'reuters'),"reut2-"+"%03d"%i+".sgm")) as input:
        self.feed(input.read())
    time_taken = time() - start_time
    logger.debug("Completed parsing in %3.2f seconds" % time_taken)
    return (self.docmap, self.classmap, self.lewissplit)


class Reuters21578(BagOfWords, ASCII):
  __name__ = 'Reuters21578'
  __parser = None

  def _parser(self):
    if self.__parser is None:
      self.__parser = ReutersParser()
      self.__parser.run()
    return self.__parser

  @property
  def instance_ids(self):
    # Overrode instance_ids to avoid having to parse the entire dataset to do dsinfo from CLI
    ids = map(str,range(1, 21579))
    if self.__parser is not None:
      # Sanity check if we have already parsed the dataset.
      assert set(self.classmap(self.classmap_names.next()).keys()) == set(ids)
    return list(sorted(ids)) 

  def ts_byte(self):
    p = self._parser()
    return p.docmap
    
  def cm_reuters21578_topics(self):
    p = self._parser()
    return p.classmap

  def sp_lewis(self):
    p = self._parser()
    return dict(train=p.lewissplit['TRAIN'], test=p.lewissplit['TEST'], unused=p.lewissplit['NOT-USED'])
