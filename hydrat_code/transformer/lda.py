from hydrat.wrapper.GibbsLDA import GibbsLDA
from hydrat.transformer import Transformer
from hydrat import config as config
from hydrat.configuration import Configurable, EXE

import hydrat.common.pb as pb

class GibbsLDATransformer(Configurable, Transformer):
  requires=\
    { ('tools','gibbslda') : EXE('lda')
    }
  def __init__(self, alpha=None, beta=0.1, niters=2000, infiters=30, ntopics=100):
    exe = config.getpath('tools','gibbslda') 
    tmp = config.getpath('paths','scratch')
    clear_temp = config.getboolean('debug','clear_temp_files')
    self.lda = GibbsLDA\
                 ( alpha=alpha
                 , beta=beta
                 , niters=niters
                 , infiters=infiters
                 , ntopics=ntopics
                 , tmp=tmp
                 , clear_temp = clear_temp
                 , exe = exe
                 )
    #TODO: Better naming!
    self.__name__ = self.__class__.__name__\
      + 'a'+str(self.lda.alpha)\
      + 'b'+str(self.lda.beta)\
      + 't'+str(self.lda.ntopics)\
      + 'e'+str(self.lda.niters)\
      + 'i'+str(self.lda.infiters)
    Transformer.__init__(self)
    
  def learn(self, feature_map, class_map):
    with pb.ProgressBar(widgets=pb.get_widget('GibbsLDA learn'), maxval=self.lda.niters) as pbar:
      self.lda.estimate(feature_map, progress_callback=pbar.update)

  def apply(self, feature_map):
    with pb.ProgressBar(widgets=pb.get_widget('GibbsLDA apply'), maxval=self.lda.infiters) as pbar:
      topics = self.lda.apply(feature_map, progress_callback=pbar.update)
    return topics
