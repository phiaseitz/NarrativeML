# Common operations on metadata stored in pytables nodes
from collections import defaultdict
from hydrat.common.counter import Counter

def value_matches(v1, v2):
  """ Define equality over all sorts of values that could end up in
      our metadata.
      ..todo:
        Would be nice to unit test this.
  """
  try:
    match = v1 == v2
    if isinstance(match, bool):
      return match
    else:
      return match.all()
  except ValueError:
    # try to catch numpy complaining about arrays being compared
    try:
      if len(v1) == len(v2):
        return all(value_matches(x,y) for x,y in zip(v1,v2))

    except TypeError:
      # can't determine lengths, so we'll just call them different
      return False

def metadata_matches(attrs, desired_metadata):
  # TODO: When do we ever need this?
  for key in desired_metadata:
    try:
      value = getattr(attrs, key)
    except AttributeError:
      # We consider node not having attribute to match if the desired value is None
      if desired_metadata[key] is not None:
        # No match if node does not have this attribute
        return False
    if not value_matches(value,desired_metadata[key]):
      # No match if node's value does not match the desired value
      return False
  # Match if we got this far
  return True

def get_metadata(node):
  metadata   = dict(    ( key, getattr(node._v_attrs, key) ) 
                    for  key
                    in   node._v_attrs._f_list()
                    )
  return metadata

def metamap(metadatas):
  """

  """
  mapping = defaultdict(Counter)
  for md in metadatas:
    for k,v in md.iteritems():
      if isinstance(v, str):
        mapping[k].update((v,))
  return mapping

def shared(*metadatas):
  """
  Compute the shared metadata of a sequence of metadata objects.
  """
  m0 = metadatas[0]
  retval = {}
  for k in m0:
    if all( m[k] == m0[k] for m in metadatas ):
      retval[k] = m0[k]
  return retval

