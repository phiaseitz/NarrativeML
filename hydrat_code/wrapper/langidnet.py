#!/usr/bin/env python
"""
Wrapper for langid.net
based on code from http://stackoverflow.com/questions/1136604/how-do-i-use-the-json-google-translate-api
"""
import urllib
import json
import time
import random
import logging
logger = logging.getLogger(__name__)

class LangidNetLangid(object):
  base_url='http://api.langid.net/identify.json?'

  def __init__( self, simulate=False, rate=None, apikey=None, retry=60 ):
    if rate is None:
      if apikey is None:
        self.rate = 100
      else:
        self.rate = 1000
    else:
      self.rate = rate
    self.simulate = simulate
    self.apikey = apikey
    self.retry = retry

  def classify(self, text):
    if isinstance(text, unicode): text = text.encode('utf-8')
    if self.simulate:
      response = {'response':{'iso':'en'}}
    else:
      query = {'string': text[:199]}
      if self.apikey is not None:
        query['api-key'] = self.apikey
      data = urllib.urlencode(query)
      search_results = urllib.urlopen(self.base_url+data)
      response = json.loads(search_results.read())
    retry = self.retry
    while response['response'] is None:
      logger.warning(response)
      logger.warning("Got a None response, retrying in %d seconds", retry)
      time.sleep(retry)
      retry *= 2
      search_results = urllib.urlopen(self.base_url+data)
      try:
        response = json.loads(search_results.read())
      except ValueError:
        response = {'response': None}
    result = response['response']['iso']
    return result

  def batch_classify(self, texts):
    result = []
    interval = 3600.0 / self.rate 
    for text in texts:
      result.append(self.classify(text))
      time.sleep(interval)
    return result
      

