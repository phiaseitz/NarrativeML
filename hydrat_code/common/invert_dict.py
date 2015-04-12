def invert_dict(dict):
  """ Inverts a mapping from a key onto a list of values, creating
  a mapping from a value to a list of keys
  """
  result = {}
  for key, values in dict.iteritems():
    for value in values:
      result[value] = result.get(value, []) + [key]
  return result
    
