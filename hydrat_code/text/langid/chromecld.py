from hydrat.text import TextClassifier
import cld
class ChromeCLD(TextClassifier):
  metadata = dict(
    class_space = 'iso639_1',
    dataset='chromeCLD',
    instance_space='chromeCLD',
    learner='chromeCLD',
    learner_params={},
    )

  def __init__(self):
    TextClassifier.__init__(self, lambda lang: 'UNKNOWN' if lang == 'un' else lang)

  def classify(self, text):
    lang = cld.detect(text)[1]
    return [lang]

