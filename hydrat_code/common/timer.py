from timeit import default_timer

class Timer(object):
  """
  Simple timing context manager. Sets an attribute "duration"
  which records the last interval timed.
  """
  def __init__(self):
    self.timer = default_timer
    self.start = None
    self.end = None

  def __enter__(self):
    self.start = self.timer()
    self.end = None
    return self

  def __exit__(self, *args):
    self.end = self.timer()

  @property
  def elapsed(self):
    now = self.timer()
    if self.end is not None:
      return self.end - self.start
    else:
      return now - self.start

  def rate(self, count):
    """
    Compute a rate based on elapsed time and count
    """
    if self.start is None:
      raise ValueError("Not yet started")

    return count / self.elapsed 
