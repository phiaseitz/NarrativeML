"""
Wrapper for interacting with the Perl TextCat Language Guesser[1] of 
Gentian Van Noord. This is provided for historical reasons; for practical
purposes libtextcat[2] is much faster, while implementing the same algorithm.
This module was renamed from textcat.py to avoid import conflicts with the
pylibtextcat binding[3].

[1] http://odur.let.rug.nl/~vannoord/TextCat/
[2] http://software.wise-guys.nl/libtextcat/
[3] http://pypi.python.org/pypi/pylibtextcat
"""
import logging
import time
import numpy

import hydrat.wrapper.textcat as textcat

from hydrat import config
from hydrat.configuration import Configurable, EXE, DIR
from hydrat.common.mapmatrix import map2matrix, matrix2map
from hydrat.common.pb import ProgressIter

logger = logging.getLogger(__name__)


# Association list for mapping textcat's internal output onto
# iso639_1
textcat2iso639_1_assoc = [
  ('afrikaans', 'af'),
  ('albanian', 'sq'),
  ('amharic-utf', 'am'),
  ('arabic-iso8859_6', 'ar'),
  ('arabic-windows1256', 'ar'),
  ('armenian', 'hy'),
  ('basque', 'eu'),
  ('belarus-windows1251', 'be'),
  ('bosnian', 'bs'),
  ('breton', 'br'),
  ('bulgarian-iso8859_5', 'bg'),
  ('catalan', 'ca'),
  ('chinese-big5', 'zh'),
  ('chinese-gb2312', 'zh'),
  ('croatian-ascii', 'hr'),
  ('czech-iso8859_2', 'cs'),
  ('danish', 'da'),
  ('dutch', 'nl'),
  ('english', 'en'),
  ('esperanto', 'eo'),
  ('estonian', 'et'),
  ('finnish', 'fi'),
  ('french', 'fr'),
  ('frisian', 'fy'),
  ('georgian', 'ka'),
  ('german', 'de'),
  ('greek-iso8859-7', 'el'),
  ('hebrew-iso8859_8', 'he'),
  ('hindi', 'hi'),
  ('hungarian', 'hu'),
  ('icelandic', 'is'),
  ('indonesian', 'id'),
  ('irish', 'ga'),
  ('italian', 'it'),
  ('japanese-euc_jp', 'ja'),
  ('japanese-shift_jis', 'ja'),
  ('korean', 'ko'),
  ('latin', 'la'),
  ('latvian', 'lv'),
  ('lithuanian', 'lt'),
  ('malay', 'ms'),
  ('manx', 'gv'),
  ('marathi', 'mr'),
  ('mingo', 'UNKNOWN'),
  ('nepali', 'ne'),
  ('norwegian', 'no'),
  ('persian', 'fa'),
  ('polish', 'pl'),
  ('portuguese', 'pt'),
  ('quechua', 'qu'),
  ('romanian', 'ro'),
  ('rumantsch', 'rm'),
  ('russian-iso8859_5', 'ru'),
  ('russian-koi8_r', 'ru'),
  ('russian-windows1251', 'ru'),
  ('sanskrit', 'sa'),
  ('scots_gaelic', 'gd'),
  ('scots', 'UNKNOWN'), #sco
  ('serbian-ascii', 'sr'),
  ('slovak-ascii', 'sk'),
  ('slovak-windows1250', 'sk'),
  ('slovenian-ascii', 'sk'),
  ('slovenian-iso8859_2', 'sk'),
  ('spanish', 'es'),
  ('swahili', 'sw'),
  ('swedish', 'sv'),
  ('tagalog', 'tl'),
  ('tamil', 'ta'),
  ('thai', 'th'),
  ('turkish', 'tr'),
  ('ukrainian-koi8_u', 'uk'),
  ('vietnamese', 'vi'),
  ('welsh', 'cy'),
  ('yiddish-utf', 'yi'),
]

textcat2iso639_1_dict = dict(textcat2iso639_1_assoc)
def textcat2iso639_1(klass):
  return [textcat2iso639_1_dict.get(k, 'UNKNOWN') for k in klass]

def identity(klass):
  return klass

class TextCat(Configurable, textcat.TextCat):
  requires =\
    { 
    ('tools', 'textcat')         : EXE('text_cat'),
    ('tools', 'textcat-models')  : DIR('LM'),
    }
  def __init__(self, classlabel_map=textcat2iso639_1):
    textcat.TextCat.__init__( self,
      config.getpath('tools','textcat'), 
      scratch=config.getpath('paths','scratch'),
      modelpath=config.getpath('tools','textcat-models'),
    )
    self.classlabel_map = classlabel_map
    # Set up our default metadata
    self.metadata = dict(
      dataset = 'textcat',
      instance_space = 'textcat', 
      tokenstream='byte', 
      class_space='iso639_1',
      train_time=None,
      learner='textcat',
      learner_params={'tokenstream':'byte','classlabel_map':self.classlabel_map.__name__},
      )

  def train(self, proxy, classlabel_map = None):
    # Replace with new training data
    ts = proxy.tokenstream
    cm = matrix2map(proxy.classmap.raw, proxy.instancelabels, proxy.classlabels)
    pairs = zip(ts, (cm[i][0] for i in proxy.instancelabels) )

    # Time the training phase
    start = time.time()
    textcat.TextCat.train(self, pairs)
    train_time = time.time() - start

    if classlabel_map is None:
      self.classlabel_map = identity 
    else:
      self.classlabel_map = classlabel_map
    # Update our new metadata
    self.metadata = dict(
      dataset = proxy.dsname,
      instance_space = proxy.instance_space, 
      tokenstream=proxy.tokenstream_name, 
      class_space=proxy.class_space,
      train_time=train_time,
      learner='textcat',
      learner_params={'tokenstream':'byte','classlabel_map':self.classlabel_map.__name__},
      )

  def classify(self, text): 
    return [ self.classlabel_map(textcat.TextCat.classify_single(self, text)) ]

  def classify_batch(self, texts, callback=None):
    retval = []
    for i, t in enumerate(texts):
      retval.append(self.classify(t))
      if callback is not None:
        callback(i)
    return retval

  def load_model(self, model, classlabel_map=None):
    """
    Load a pre-trained model.
    """
    self.model = model

    if classlabel_map is None:
      self.classlabel_map = identity 
    else:
      self.classlabel_map = classlabel_map


from hydrat.datamodel import TaskSetResult, Result
class TextCatExperiment(TaskSetResult):
  def __init__(self, cl_proxy, train_proxy=None, tokenstream='byte'):
    self.cl_proxy = cl_proxy
    self.train_proxy = train_proxy
    self.classifier = TextCat()
    if train_proxy is not None:
      self.classifier.train(train_proxy)

  @property
  def metadata(self):
    md = dict(\
      class_space  = self.classifier.metadata['class_space'],
      dataset      = self.classifier.metadata['dataset'],
      instance_space = self.classifier.metadata['instance_space'],
      eval_dataset = self.cl_proxy.dsname,
      eval_space   = self.cl_proxy.instance_space,
      learner      = 'textcat',
      learner_params = dict(
        tokenstream=self.classifier.metadata['tokenstream'], 
        classlabel_map=self.classifier.classlabel_map.__name__,
        ),
      )
    return md

  @property
  def results(self):
    proxy = self.cl_proxy
    ts = proxy.tokenstream

    start = time.time()
    ids = proxy.instancelabels
    texts = proxy.tokenstream
    class_map = {}
    for id, text in ProgressIter(zip(ids, texts),label='TextCat'):
      class_map[id] = [ self.classifier.classify(text) ]
    test_time = time.time() - start
    
    cl = map2matrix( class_map, proxy.instancelabels, proxy.classlabels )
    gs = proxy.classmap.raw

    result_md = dict()
    result_md['learn_time'] = self.classifier.metadata['train_time']
    result_md['classify_time'] = test_time
    instance_indices = numpy.arange(len(proxy.instancelabels))
    return [ Result(gs, cl, instance_indices, result_md ) ]

class TextCatCrossvalidate(TaskSetResult):
  def __init__(self, proxy):
    self.classifier = TextCat()
    self.proxy = proxy

  @property
  def metadata(self):
    return dict(\
      class_space  = self.proxy.class_space,
      dataset      = self.proxy.dsname,
      instance_space = self.proxy.instance_space,
      learner      = 'textcat',
      learner_params = dict(tokenstream='byte'),
      )

  @property
  def results(self):
    # Should refactor this against TextCatExperiment, there is so much in common.
    raise NotImplementedError, "Need to finish cleaning this up"
    results = []
    for i in ProgressIter(range(num_fold), label="TextCat Crossvalidation"):
      # train a textcat instance
      train_ids = instance_ids[split[:,i,0]]
      pairs = [ (ts[id], cm[id][0]) for id in train_ids ]
      self.classifier.train(pairs)

      # run the test data against it
      test_ids = instance_ids[split[:,i,1]]

      class_map = dict(zip(test_ids, cat.classify([ts[id] for id in test_ids])))

      result_md = dict(md)
      result_md['learn_time'] = cat.__timing_data__['train']
      result_md['classify_time'] = cat.__timing_data__['classify']

      instance_indices = membership_vector(test_ids, ds.instance_ids)
      cl = map2matrix( class_map, test_ids, classlabels )
      gs = map2matrix( cm, test_ids, classlabels )
      results.append(Result(gs, cl, instance_indices, result_md))
    return results
