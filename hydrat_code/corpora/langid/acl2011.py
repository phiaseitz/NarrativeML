from hydrat import config
from hydrat.dataset.iso639 import ISO639_1
from hydrat.dataset.text import ByteUBT, SingleDir
from hydrat.dataset.encoded import CodepointUBT, UTF8
from hydrat.dataset.split import CrossValidation
from hydrat.configuration import Configurable, DIR

class Wiki10k(Configurable, ISO639_1, UTF8, CrossValidation, ByteUBT, CodepointUBT, SingleDir):
  requires={ ('corpora', 'acl2011-langid-wiki10k') : DIR('wiki10k') }

  def data_path(self):
    return config.getpath('corpora', 'acl2011-langid-wiki10k')

  def cm_iso639_1(self):
    ids = self.tokenstream('byte').keys()
    return dict( (i,[i.split('_')[0]]) for i in ids )

  def sp_crossvalidation(self):
    from numpy.random.mtrand import RandomState
    return self.crossvalidation('iso639_1', 10, RandomState(61383441363))
