class Inconsistent(Exception): pass

class enumerator():
  """Class providing handly enumeration facilities"""

  def __init__(self):
    self.idlabeldict = {}
    self.labeliddict = {}
    self.counter = 0

  def merge(self, enum):
    """Merge two enumerators, checking them for consistency"""
    for key in enum.labeliddict:
      if key in self.labeliddict:
        if self.labeliddict[key] != enum.labeliddict[key]:
          raise Inconsistent, "Unable to merge enums due to inconsistency for key: %s" % str(key)
    self.labeliddict.update(enum.labeliddict)
    self.idlabeldict.update(enum.idlabeldict)
    self.counter = len(self.idlabeldict) + 1

  def append(self, enum):
    """Append another enumerator into the current one"""
    for key in enum.labeliddict:
      if key in self.labeliddict:
        raise Inconsistent, "Unable to append enums as key is present in both: %s" % str(key)
    for key in sorted(enum.idlabeldict):
      id    = self.counter
      label = enum.idlabeldict[key]
      self.idlabeldict[id]     = label
      self.labeliddict[label]  = id
      self.counter += 1

      
  def assign(self, label):
    try:
      id = self.labeliddict[label]
    except KeyError:
      id = self.counter
      self.idlabeldict[id] = label
      self.labeliddict[label] = id
      self.counter += 1
    return id

  def lookup(self, id):
    return self.idlabeldict[id]

  def lookupLabel(self, label):
    return self.labeliddict[label]

  def numLabels(self):
    return self.counter 

  def __getinitargs__(self):
    return ()

  def __setstate__(self, (ild, lid) ):
    self.idlabeldict.update(ild)
    self.labeliddict.update(lid)
    assert len(self.idlabeldict) == len(self.labeliddict), "enumerator load error: dict sizes do not match"
    self.counter = len(self.idlabeldict)

  def __getstate__(self):
    return (self.idlabeldict, self.labeliddict)
