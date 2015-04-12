from hydrat.text import TextClassifier
import textcat
class LibTextCat(TextClassifier):
  metadata = dict(
    class_space = 'iso639_1',
    dataset='libtextcat',
    instance_space='libtextcat',
    learner='libtextcat',
    learner_params={},
    )

  def __init__(self, config, base=None):
    TextClassifier.__init__(self, None) # TODO output map
    self.config = config
    self.base = base
    self.textcat = textcat.TextCat(config, base)
    self.metadata['learner_params']['config'] = config
    self.metadata['learner_params']['base'] = base

  def classify(self, text):
    try:
      return self.textcat.classify(text)
    except (textcat.UnknownException, textcat.ShortException):
      return ['UNKNOWN']
