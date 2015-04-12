"""
Wrapper for language-detection
http://code.google.com/p/language-detection/
"""
import tempfile
import os
import re
import shutil
import itertools
import operator

from subprocess import Popen, PIPE, STDOUT

# /tmp/LangDetect-_PbHSq:[en:0.9999970955345286]
# /tmp/LangDetect-4QdM8R:[de:0.5714246034071901, af:0.4285733847851334]
RE_LANGDETECT_OUTPUT = re.compile(r'(?P<file>[/\.\w-]+):\[(?P<class>[-\w]+):\d\.\d+(, [-\w]+:\d\.\d+|)+\]')

class LangDetect(object):
  def __init__(self, javapath, toolpath, profilespath, scratch, batchsize = 100):
    self.profilespath = profilespath
    self.javapath = javapath
    self.toolpath = toolpath
    self.scratch = scratch
    self.batchsize = batchsize

  def train(self, pairs):
    # Training is supported by the package, but the package also ships with premade profiles
    # so we use it as an off-the-shelf reference
    raise NotImplementedError

  def classify(self, text):
    # java -jar lib/langdetect.jar --detectlang -d [profile directory] [test file(s)]
    testfile = tempfile.NamedTemporaryFile(dir=self.scratch, prefix='LangDetect-')
    testfile.write(text)
    testfile.flush()
    p = Popen(\
          [ self.javapath, '-jar', self.toolpath, '--detectlang', '-d', self.profilespath, testfile.name ],
          stdout=PIPE,
          stdin=None,
          stderr=STDOUT,
          )
    out = p.communicate()[0]
    if p.returncode == 0:
      return out.split(':')[1][1:]
    raise ValueError, "library returned retcode: %d" % p.returncode

  def classify_batch(self, texts, callback=None):
    retval = []
    cm = {}
    t_iter = iter(texts)
    while True:
      batch = itertools.islice(t_iter, self.batchsize)
      testfiles = []
      for text in batch:
        testfile = tempfile.NamedTemporaryFile(dir=self.scratch, prefix='LangDetect-')
        testfile.write(text)
        testfile.flush()
        testfiles.append(testfile)
      if len(testfiles) == 0:
        # No more files
        break
      tf_names = [t.name for t in testfiles]
      p = Popen(\
            [ self.javapath, '-jar', self.toolpath, '--detectlang', '-d', self.profilespath, ] + tf_names,
            stdout=PIPE,
            stdin=None,
            stderr=STDOUT,
            )
      out = p.communicate()[0]


      # Add the new results into the classmap
      if p.returncode == 0:
        if out.startswith("ERROR: Not found profile:"):
          raise ValueError, out
        for row in out.split('\n'):
          match = RE_LANGDETECT_OUTPUT.match(row)
          if match:
            cm[match.group('file')] = match.group('class')
      else:
        raise ValueError, "library returned retcode: %d" % p.returncode
      
      retval.extend(cm.get(name, 'UNKNOWN') for name in tf_names)

      if callback is not None:
        callback(len(retval))

      
    return retval
