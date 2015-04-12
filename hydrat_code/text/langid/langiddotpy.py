from hydrat.text import TextClassifier
import langid

class LangidDotPy(TextClassifier):
  metadata = dict(
    class_space = 'iso639_1',
    dataset='ijcnlp2011',
    instance_space='langid.py',
    learner='langid.py',
    learner_params={},
    )

  def classify(self, text):
    return [langid.classify(text)[0]]

