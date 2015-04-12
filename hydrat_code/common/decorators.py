import os
import logging
import warnings
import functools
import inspect
import time
from hydrat import config


#from http://wiki.python.org/moin/PythonDecoratorLibrary
class memoized(object):
   """Decorator that caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned, and
   not re-evaluated.
   """
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      try:
         return self.cache[args]
      except KeyError:
         self.cache[args] = value = self.func(*args)
         return value
      except TypeError:
         # uncachable -- for instance, passing a list as an argument.
         # Better to not cache than to blow up entirely.
         return self.func(*args)
   def __repr__(self):
      """Return the function's docstring."""
      return self.func.__doc__

class shelved(object):
  """ Decorator maker for shelved """
  def __init__(self, path):
    scratchpath = config.get('paths','scratch')
    shelfpath = os.path.join(scratchpath, 'shelves')
    self.path = os.path.join(shelfpath, path)
    
  def __call__(self, func):
    return Shelved(self.path, func)

class Shelved(object):
   """Decorator that caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned, and
   not re-evaluated. Also shelves the result in the specified file for future
   use.
   """
   def __init__(self, shelf_path, func):
      self.func = func
      self.cache_loaded = False
      self.shelf_path = shelf_path
      self.logger = logging.getLogger('hydrat.common.decorators.Shelved')

   def __call__(self, *args):
      if not self.cache_loaded:
        self.logger.info("Loading shelved data from: %s", self.shelf_path)
        try:
          shelf_file = open(self.shelf_path,"r")
          from cPickle import load
          try:
            self.cache = load(shelf_file)
          except EOFError:
            self.logger.warning("%s is corrupted!", self.shelf_path)
            self.cache = {} 
          shelf_file.close()
        except IOError:
          self.logger.info("No exisiting cachefile")
          self.cache = {} 
        self.logger.debug("Initial cache size: %d", len(self.cache))
        self.cache_loaded = True
      try:
        return self.cache[args]
      except KeyError:
        self.logger.debug("Not in cache: %s", str(args))
        self.cache[args] = value = self.func(*args)
        from cPickle import dump
        file = open(self.shelf_path, "w")
        dump(self.cache, file)
        file.close()
        return value
      except TypeError:
        # uncachable -- for instance, passing a list as an argument.
        # Better to not cache than to blow up entirely.
        self.logger.warning("Failed to cache")
        return self.func(*args)
   def __repr__(self):
      """Return the function's docstring."""
      return self.func.__doc__

class replace_with_result(object):
    def __init__ (self, fn):
        self.fn = fn
        self.__name__ = fn.__name__

    def __call__ (self, instance = None):
        if instance and self.instance:
          raise TypeError, "Bound %s() requires no arguments" % self.__name__
        if instance:
          return self.fn(instance)
        else :
          result =  self.fn(self.instance)
          setattr(self.instance, self.fn.__name__, lambda : result)
          return result

    def __get__(self, instance, objtype = None):
        self.instance = instance
        return self

    def __repr__(self):
        """Return the function's docstring."""
        return self.fn.__doc__ + " (cached)"


class timed(object):
  """ Decorator that adds execution time of a method to a
      dictionary called __timing_data__
  """
  def __init__(self, decoratee):
    self.decoratee = decoratee
    self.__name__ = decoratee.__name__
    self.instance = None
    self.owner = None

  def __get__(self, instance, owner):
    self.instance = instance
    self.owner = owner
    return self

  def __call__(self, *args, **kwargs):
    if self.owner is None:
      # If we have no owner, we are decorating a function
      # TODO: Can we inject into a global in this case?
      raise NotImplementedError, "Only implemented for methods"

    if self.instance:
      # If we have an instance we are bound
      instance = self.instance
    else:
      # Otherwise, we are unbound and the first argument is the instance
      instance = args[0]
      args = args[1:]

    # Do the actual operation
    start = time.time()
    retval = self.decoratee(instance, *args, **kwargs)
    duration = time.time() - start

    # Create the __timing_data__ dictionary if needed
    if not hasattr(instance, '__timing_data__'):
      instance.__timing_data__ = {}

    # Attach the measured time
    instance.__timing_data__[self.__name__] = duration

    return retval




# from http://wiki.python.org/moin/PythonDecoratorLibrary#GeneratingDeprecationWarnings
def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        frame = inspect.currentframe()
        warnings.warn_explicit(
            "Call to deprecated function %(funcname)s." % {
                'funcname': func.__name__,
            },
            category=DeprecationWarning,
            #filename=func.func_code.co_filename,
            #lineno=func.func_code.co_firstlineno + 1
            filename = frame.f_back.f_code.co_filename,
            lineno = frame.f_back.f_lineno
        )

        return func(*args, **kwargs)
    return new_func

# from http://wiki.python.org/moin/PythonDecoratorLibrary#Synchronization
def synchronized(lock):
    """ Synchronization decorator. """

    def wrap(f):
        def new_function(*args, **kw):
            lock.acquire()
            try:
                return f(*args, **kw)
            finally:
                lock.release()
        return new_function
    return wrap
