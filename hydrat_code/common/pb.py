import progressbar as pb
ENABLED = True

def get_widget(label):
  return [ label, ' ', Proportion(), ' ', pb.Bar(marker='>',left='[',right=']'), ' ', pb.ETA() ]

class Proportion(pb.ProgressBarWidget):
    "Proportion done."
    def update(self, pbar):
        return '%d/%d' % (pbar.currval, pbar.maxval)

class ProgressBar(pb.ProgressBar):
  def __enter__(self):
    self.start()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.finish()

  def _need_update(self):
    # Always update, since we mostly use this on slow-running tasks
    return True

  def update(self, value):
    if ENABLED:
      pb.ProgressBar.update(self, value)

class ProgressIter(object):
  def __init__(self, sequence, label='Progress', maxval=None):
    self.seq = iter(sequence)
    maxval = len(sequence) if maxval is None else maxval
    maxval = maxval if maxval > 0 else 1 #handle empty sequences
    self.pb = ProgressBar(widgets=get_widget(label),maxval=maxval)
    self.pb.start()
    self.count = 0

  def __iter__(self):
    return self

  def next(self):
    try:
      next = self.seq.next()
      self.pb.update(self.count)
      self.count += 1
      return next
    except StopIteration,e :
      self.pb.finish()
      raise e
    
