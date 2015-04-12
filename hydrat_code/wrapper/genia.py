"""
Tokenstream generator based on GENIA tagger. 
http://www-tsujii.is.s.u-tokyo.ac.jp/GENIA/home/wiki.cgi

This wrapper around GENIA manages a single GENIA instance, allowing
for batch tagging without incurring the start-up time for each 
instance being tagged.
"""
import subprocess
import re
from collections import namedtuple

RE_NEWLINE = re.compile(r'\n')
GENIA_TOKENFIELDS = ['word', 'base', 'POStag', 'chunktag', 'NEtag']

GeniaToken = namedtuple('GeniaToken', GENIA_TOKENFIELDS + ['start', 'end'])

class GeniaTagger(object):
  def __init__(self, tagger_exe=None, genia_path=None):
    # TODO: Handle paths not specified
    self.genia_instance = subprocess.Popen([tagger_exe], cwd=genia_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Close stderr to avoid deadlocks due to a full buffer.
    self.genia_instance.stderr.close()

  def __del__(self):
    self.genia_instance.terminate()

  def process(self, text, find_span=True):
    # Strip off newlines, as genia uses them to delimit blocks to process
    proc_text = RE_NEWLINE.sub('', text)
    # Write the text to genia's stdin, terminating with a newline
    self.genia_instance.stdin.write(proc_text + '\n')
    token_stream = []
    # Read all the lines on genia's stdout. Output ends with a blank line.
    line = self.genia_instance.stdout.readline().rstrip()
    range_start = 0
    while line:
      data = line.split('\t')

      # Extract the raw word
      word = data[0]
      # Clean up genia's mangling of some tokens.
      if word == '``': word = '"'
      if word == "''": word = '"'
      # Rewrite cleaned token
      data[0] = word

      if find_span:
        # Compute where this token starts and ends in the stream
        start, end = re.search(re.escape(word), proc_text[range_start:]).span()
        token_start = range_start + start
        token_end = range_start + end
        range_start = range_start + end
      else:
        token_start, token_end = 0,0

      token = GeniaToken._make(data+[token_start,token_end])
      token_stream.append(token)
      line = self.genia_instance.stdout.readline().rstrip()
    return token_stream 


