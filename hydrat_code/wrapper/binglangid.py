#!/usr/bin/env python
"""
Wrapper for bing's langid API
based on code from http://stackoverflow.com/questions/1136604/how-do-i-use-the-json-google-translate-api
and adapted from the Google langid API wrapper

Marco Lui <saffsd@gmail.com>
December 2011
"""
import urllib, urllib2
import httplib
import time
import random
import logging
import xml.etree.ElementTree as etree
logger = logging.getLogger(__name__)

class BingLangid(object):
  base_url='http://api.microsofttranslator.com/v2/Http.svc/Detect'

  def __init__( self, apikey, sleep=5, simulate=False, retry = 60, referer=None):
    self.sleep = sleep # Time between requests
    self.simulate = simulate
    self.retry = retry
    self.apikey = apikey
    self.referer = referer

  def classify(self, text):
    if isinstance(text, unicode): text = text.encode('utf-8')
    if self.simulate:
      response = '<string xmlns="http://schemas.microsoft.com/2003/10/Serialization/">en</string>'
    else:
      query = {'appId':self.apikey,'text':text}
      data = urllib.urlencode(query)
      req = urllib2.Request(self.base_url+'?'+data)
      logger.debug("URL:%s", req.get_full_url())
      if self.referer is not None: req.add_header('Referer',self.referer)
      try:
        search_results = urllib2.urlopen(req)
        response = search_results.read()
      except (urllib2.URLError, urllib2.HTTPError, httplib.BadStatusLine),e:
        logger.warning(e)
        response = ''
        

    retry = self.retry
    while not response:
      logger.warning(response)
      logger.warning("retrying in %d seconds", retry)
      time.sleep(retry)
      retry *= 2
      try:
        search_results = urllib2.urlopen(req)
        response = search_results.read()
      except (ValueError, urllib2.URLError, urllib2.HTTPError),e:
        logger.warning(e)
        logger.warning("response: %s", response)
        response = ''

    response_node = etree.fromstring(response)
    return response_node.text

  def batch_classify(self, texts):
    for text in texts:
      yield self.classify(text)
      time.sleep(random.random() * self.sleep)
      

