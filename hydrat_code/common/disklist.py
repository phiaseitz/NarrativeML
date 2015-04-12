"""
Disk-backed queue. Not to be used for object persistence.
Items can be added to the queue, and the queue can be iterated,

"""

import collections
import marshal
import tempfile

class disklist(collections.Iterable, collections.Sized):
  def __init__(self, temp_dir=None):
    self.fileh = tempfile.TemporaryFile(dir=temp_dir, suffix='-disklist')
    self.count = 0

  def __iter__(self):
    self.fileh.seek(0)
    while True:
      try:
        yield marshal.load(self.fileh)
      except (EOFError, ValueError, TypeError):
        break

  def __len__(self):
    return self.count

  def append(self, value):
    marshal.dump(value, self.fileh)
    self.count += 1

if __name__ == "__main__":
  x = disklist()
  x.append(1)
  x.append(2)
  x.append(3)
  print list(x)
  x.append(3)
  print list(x)
  x.append(4)
  print list(x)
  print list(x)
