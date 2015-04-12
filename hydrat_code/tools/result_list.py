#!/usr/bin/env python
import sys
from optparse import OptionParser

from hydrat.result.resulttable import ResultTableReader
from hydrat.tools import parse_metatags

if __name__ == "__main__":
  usage = "usage: %prog [options] filename"
  parser = OptionParser(usage)
  parser.add_option("-m", action='append', dest='metatags', help="Match metadata KEY:VALUE")
  parser.add_option("-l", action='append', dest='metadata', help="Print all metadata values for KEY")

  options, args = parser.parse_args()
  if len(args) != 1:
    parser.error("incorrect number of arguments")

  result_path = args[0]

  try:
    desired_metadata = parse_metatags(options.metatags)
  except ValueError:
    parser.error("Invalid metadata")

  reader = ResultTableReader(result_path)

  print result_path, "contains", len(reader), "results"

  if options.metadata is not None:
    metadata_map = reader.get_metadata_map()
    for key in options.metadata:
      if key in metadata_map:
        print key
        print "  " + str(metadata_map[key].keys())
      else:
        print key, "is not a known metadata key"

  if desired_metadata != {}:
    resolved_tags = reader.resolve_tag(desired_metadata)
    print len(resolved_tags), "results match requested tags"
    for tag in resolved_tags:
      print reader[tag]
      print
       
  reader.close()





   

