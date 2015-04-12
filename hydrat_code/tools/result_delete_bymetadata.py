#!/usr/bin/env python
import sys
from optparse import OptionParser

from hydrat.result.resulttable import ResultTableWriter

if __name__ == "__main__":
  usage = "usage: %prog [options] filename"
  parser = OptionParser(usage)
  parser.add_option("-m", action='append', dest='metatags', help="Match metadata KEY:VALUE")

  options, args = parser.parse_args()
  if len(args) != 1:
    parser.error("incorrect number of arguments")

  result_path = args[0]

  desired_metadata = {}
  if options.metatags is None:
    parser.error("Must provide some tags to trash by")
  else:
    for metatag in options.metatags:
      try:
        key,value = metatag.split(':')
      except ValueError:
        parser.error("Invalid metatag: %s" % metatag)
      desired_metadata[key] = value

  reader = ResultTableWriter(result_path)

  print result_path, "contains", len(reader), "results"
  resolved_tags = reader.resolve_tag(desired_metadata)
  print len(resolved_tags), "results match requested tags"
  trash = raw_input("Are you sure you wish to trash them?")

  if trash == 'y':
    for tag in resolved_tags:
      print "Trashing", tag
      reader.del_TaskSetResult(tag)
  else:
    print 'aborted'
    
  reader.close()





   

