#!/usr/bin/env python
"""
Wrapper for google's langid API
based on code from http://stackoverflow.com/questions/1136604/how-do-i-use-the-json-google-translate-api
"""
import urllib, urllib2
import httplib
import json
import time
import random
import logging
logger = logging.getLogger(__name__)

class GoogleLangid(object):
  base_url='http://ajax.googleapis.com/ajax/services/language/detect'

  def __init__( self, sleep=5, simulate=False, retry = 60, apikey=None, referer=None, confidence=False ):
    self.sleep = sleep # Time between requests
    self.simulate = simulate
    self.retry = retry
    self.apikey = apikey
    self.referer = referer
    self.confidence = confidence

  def classify(self, text):
    if isinstance(text, unicode): text = text.encode('utf-8')
    if self.simulate:
      response = {'responseData':{'language':'en'}}
    else:
      query = {'v':1.0,'ie': 'UTF8', 'q': text}
      if self.apikey is not None:
        query['key'] = self.apikey
      data = urllib.urlencode(query)
      req = urllib2.Request(self.base_url+'?'+data)
      if self.referer is not None: req.add_header('Referer',self.referer)
      try:
        search_results = urllib2.urlopen(req)
        response = json.loads(search_results.read())
      except (urllib2.URLError, httplib.BadStatusLine):
        response = {'responseData':None}
        

      #search_results = urllib.urlopen(self.base_url+data)
    retry = self.retry
    while response['responseData'] is None:
      logger.warning(response)
      logger.warning("retrying in %d seconds", retry)
      time.sleep(retry)
      retry *= 2
      search_results = urllib2.urlopen(req)
      try:
        response = json.loads(search_results.read())
      except (ValueError, urllib2.URLError),e:
        logger.warning(response,e)
        response = {'responseData': None}
    result = response['responseData']['language']
    confidence = float(response['responseData']['confidence'])
    if self.confidence:
      return result, confidence
    else:
      return result

  def batch_classify(self, texts):
    for text in texts:
      yield self.classify(text)
      time.sleep(random.random() * self.sleep)
      

