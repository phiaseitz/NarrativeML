"""
Dataset implementation for langid dataset based on Debian
i18n.

Marco Lui Feb 2011
"""
import os
import csv
import codecs

from collections import defaultdict 

from hydrat import config
from hydrat.dataset.iso639 import ISO639_1, ISO639_1_CODES
from hydrat.dataset.text import ByteUBT, DirPerClass
from hydrat.dataset.encoded import CodepointUBT
from hydrat.common.decorators import replace_with_result
from hydrat.configuration import Configurable, DIR

class DebianPOFiles(Configurable, ISO639_1, ByteUBT, CodepointUBT):
  threshold = 10 
  requires={
    ('corpora', 'debian-pofiles') : DIR('debian-pofiles'),
    }

  def data_path(self):
    return os.path.join(config.getpath('corpora', 'debian-pofiles'), 'debian-unstable-po-v1', 'extracted')

  @replace_with_result
  def cm_iso639_1(self):
    path = self.data_path()
    codes = set(ISO639_1_CODES)
    cm = dict()
    dist = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(path):
      if dirpath == path: 
        # In the root directory
        dirnames[:] = [ f for f in dirnames if f in codes ]
      elif os.path.split(dirpath)[0] == path:
        # In the directory for each language
        base, lang = os.path.split(dirpath)
        for filename in filenames:
          doc_id = '%s-%s' % (filename, lang)
          dist[lang].append(doc_id)
          cm[doc_id] = [lang]
      else:
        # How did we get here?
        break

    # remove classes with less than 'threshold' documents
    for key in dist:
      if len(dist[key]) < self.threshold:
        for doc_id in dist[key]:
          del cm[doc_id]
    return cm

  @replace_with_result
  def ts_byte(self):
    path = self.data_path()
    ids = self.instance_ids
    ts = dict()
    for id in ids:
      filename, lang = id.split('-')
      with open(os.path.join(path, lang, filename)) as file:
        ts[id] = file.read()
    return ts

  def encodings(self):
    path = self.data_path()
    ids = set(self.instance_ids)
    encodings = dict()

    with open(os.path.join(path, 'encodings')) as enc_file:
      reader = csv.reader(enc_file, delimiter='\t')
      for filename, lang, enc in reader:
        doc_id = '%s-%s' % (filename, lang)
        if doc_id in ids:
          # TODO: push this codecs check further up
          try:
            codecs.lookup(enc)
          except LookupError:
            raise ValueError, "unknown encoding %s for %s" % (enc, doc_id)
          encodings[doc_id] = enc

      # We will pretend all of 'en' is utf-8 and hope for the best!
      for doc_id in ids:
        if doc_id not in encodings:
          encodings[doc_id] = 'utf-8'
    return encodings

class DebianLang10k(Configurable, DirPerClass, ISO639_1, ByteUBT, CodepointUBT):
  requires={
    ('corpora', 'debian-pofiles') : DIR('debian-pofiles'),
    }

  def data_path(self):
    return os.path.join(config.getpath('corpora', 'debian-pofiles'), 'debian-unstable-po-v2', 'lang')

  def cm_iso639_1(self):
    return self.classmap('dirname')

  def encodings(self):
    path = self.data_path()
    ids = set(self.instance_ids)
    encodings = dict()

    with open(os.path.join(path, 'metadata')) as enc_file:
      reader = csv.reader(enc_file)
      for doc_id, lang, enc in reader:
        if doc_id in ids:
          # TODO: push this codecs check further up
          try:
            codecs.lookup(enc)
          except LookupError:
            raise ValueError, "unknown encoding %s for %s" % (enc, doc_id)
          encodings[doc_id] = enc

      # We will pretend all of 'en' is utf-8 and hope for the best!
      for doc_id in ids:
        if doc_id not in encodings:
          raise ValueError, "%s does not have an encoding" % doc_id
    return encodings

class DebianDomain10k(DebianLang10k):
  def data_path(self):
    return os.path.join(config.getpath('corpora', 'debian-pofiles'), 'debian-unstable-po-v2', 'domain')
