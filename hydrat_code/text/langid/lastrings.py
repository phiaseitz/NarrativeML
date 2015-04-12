from hydrat.text import TextClassifier
from hydrat import config
from hydrat.configuration import Configurable, DIR, EXE

import subprocess
import tempfile
import os
import csv
from contextlib import closing
from collections import Counter
class LAStrings(Configurable, TextClassifier):
  """
  Wrapper for la-strings
  http://sourceforge.net/projects/la-strings/
  """
  requires={
    ('tools','lastrings')        : EXE('la-strings'),
    ('tools','lastrings-data')   : DIR('LAStrings-1.18'),
    }

  metadata = dict(
    class_space = 'iso639_1',
    dataset='lastrings',
    instance_space='lastrings',
    learner='lastrings',
    learner_params={},
    )

  def __init__(self):
    self.tooldir   = config.getpath('tools','lastrings-data')
    self.toolpath  = config.getpath('tools','lastrings')
    self.tempdir   = config.getpath('paths','scratch')
    TextClassifier.__init__(self, lambda l: l if len(l) == 2 else 'UNKNOWN')

  def classify(self, text):
    with tempfile.NamedTemporaryFile(delete=False, dir=self.tempdir) as f:
      f.write(text)
    p = subprocess.Popen([self.toolpath,'-i','-I','1',f.name], cwd=self.tooldir, stdout=subprocess.PIPE)

    preds = Counter(row.split('\t',1)[0] for row in p.stdout)
    os.unlink(f.name)
    retval =  [ l for l, freq in preds.most_common(1) ]
    return retval

