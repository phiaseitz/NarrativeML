#!/usr/bin/env python
from optparse import OptionParser

import hydrat.result.resulttable as rt
from hydrat.common.log import getHydraLogger

if __name__ == '__main__':
  logger = getHydraLogger()
  usage = "usage: %prog [options] filename"
  parser = OptionParser(usage)
  
  options, args = parser.parse_args()
  if len(args) != 2:
    parser.error("incorrect number of arguments")

  src_path = args[0]
  dst_path = args[1]

  src = rt.ResultTableReader(src_path)
  dst = rt.ResultTableWriter(dst_path)

  print "Merging %s into %s" % (src_path, dst_path)
  rt.merge(src, dst)

  src.close()
  dst.close()

