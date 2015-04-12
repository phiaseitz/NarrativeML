"""
Wrapper for the famous TextCat
http://www.let.rug.nl/~vannoord/TextCat/text_cat.tgz
"""
import tempfile
import os
import shutil
import itertools
import operator
import time

from subprocess import Popen, PIPE, STDOUT

class TextCat(object):
  def __init__(self, toolpath, scratch='/tmp', modelpath=None):
    self.toolpath = toolpath
    self.scratch = scratch
    self.model_path = modelpath
    self.trained = False

  def __del__(self):
    # Must check if we trained this model to avoid deleting external models
    if self.trained and self.model_path is not None:
      shutil.rmtree(self.model_path)

  def train(self, pairs):
    # Delete old trained model if any
    if self.trained:
      shutil.rmtree(self.model_path)
      
    # Create a temporary directory to store models
    self.model_path = tempfile.mkdtemp(prefix='textcat', dir=self.scratch)

    key = operator.itemgetter(1)
    for klass, group in itertools.groupby(sorted(pairs,key=key),key):
      class_path = os.path.join(self.model_path, klass + '.lm')
      class_data = '\n'.join(t for t,c in group)
      p = Popen([self.toolpath, '-n'], stdout=open(class_path,'w'), stdin=PIPE, stderr=STDOUT)
      p.communicate(input=class_data)[0]

    # Mark as trained
    self.trained = True

  @property
  def model(self):
    if self.model_path == None:
      raise ValueError, "no models"

    model = {}
    for key in os.listdir(self.model_path):
      with open(os.path.join(self.model_path, key)) as f:
        model[key] = f.read()

    return (self.metadata, model)

  @model.setter
  def model(self, model):
    # Separate model and metadata
    metadata, model = model

    # Delete old trained model if any
    if self.trained:
      shutil.rmtree(self.model_path)

    # Create a temporary directory to store models
    self.model_path = tempfile.mkdtemp(prefix='textcat', dir=self.scratch)

    # Output the model
    for key in model:
      with open(os.path.join(self.model_path, key), 'w') as f:
        f.write(model[key])

    # Mark as trained
    self.trained = True

    # Update metadata
    self.metadata.update(metadata)



  def classify_single(self, text):
    p = Popen\
          ( [self.toolpath, '-u1', '-d%s'%self.model_path]
          , stdout=PIPE
          , stdin=PIPE
          , stderr=STDOUT
          )
    out = p.communicate(input=text)[0]
    return out.strip()

  def batch_classify(self, texts):
    return map(self.classify_single, texts)

