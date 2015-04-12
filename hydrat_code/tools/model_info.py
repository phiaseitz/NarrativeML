#!/usr/bin/env python
# TODO: Merge this into the CLI tool!
import sys
from optparse import OptionParser

from hydrat.tools import parse_metatags
from hydrat.store import Store

if __name__ == "__main__":
  usage = "usage: %prog [options] filename"
  parser = OptionParser(usage)
  #parser.add_option("-m", action='append', dest='metatags', help="Match metadata KEY:VALUE")
  parser.add_option("-v", action='store_true', dest='verbose', help='Verbose output')

  options, args = parser.parse_args()
  if len(args) == 0:
    path = None
  elif len(args) == 1:
    path = args[0]
  else: 
    parser.error("Incorrect number of arguments")

  store = Store(path) 

  # List datasets
  print store
  ds_list = store.list_Datasets()
  print "DATASETS: %d" % len(ds_list)
  if options.verbose:
    for ds in ds_list:
      print store.resolve_Dataset(ds), ds

  # List class spaces
  cl_list = store.list_ClassSpaces()
  print "CLASS SPACES: %d" % len(cl_list)
  if options.verbose:
    for s in cl_list:
      print store.resolve_Space({'type':'class', 'name':s}), s

  # List feature spaces
  f_list = store.list_FeatureSpaces()
  print "FEATURE SPACES: %d" % len(f_list)
  if options.verbose:
    for s in f_list:
      print store.resolve_Space({'type':'feature', 'name':s}), s

  # List tasksets
  tags = store._resolve_TaskSet({})
  print "TASKSETS: %s" % len(tags)
  if options.verbose:
    for t in tags:
      task = store._get_TaskSet(t)
      print task
  
  # List results
  tags = store._resolve_TaskSetResults({})
  print "RESULTS: %d" % len(tags)
  if options.verbose:
    for t in tags:
      res = store._get_TaskSetResult(t)
      print res
  
 
  
  
  
