def parse_metatags(metatags, separator=':'):
  desired_metadata = {}
  if metatags is not None:
    for metatag in metatags:
      try:
        key,value = metatag.split(separator)
      except ValueError:
        raise ValueError, ("Invalid metatag: %s" % metatag)
      desired_metadata[key] = value
  return desired_metadata
