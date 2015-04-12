# experiment.py
# Marco Lui Feb 2011
#
# This class represents an experiment. Its intention is to act as the interface between the user
# and the udnerlying store at the level of managing tasks and results. The dataset abstraction is
# delegated to the DataProxy object.

from hydrat import config
from hydrat.datamodel import TaskSetResult, Result, BasicTask
from hydrat.common.pb import ProgressIter
import cPickle
import multiprocessing as mp

class ExperimentFold(object):
  def __init__(self, task, learner, add_args={}):
    self.task = task
    self.learner = learner
    self.add_args = add_args

  def __getstate__(self):
    # TODO: Potentially use a disk-backed implementation of tasks
    task = BasicTask.from_task(self.task)
    return {'task':task, 'learner':self.learner, 'add_args':self.add_args}

  @property
  def classifier(self):
    train_add_args = dict( (k, v[self.task.train_indices]) for k,v in self.add_args.items())
    classifier = self.learner( self.task.train_vectors, self.task.train_classes,\
        sequence=self.task.train_sequence, indices=self.task.train_indices, **train_add_args)
    return classifier

  @property
  def result(self):
    classifier = self.classifier
    test_add_args = dict( (k, v[self.task.test_indices]) for k,v in self.add_args.items())
    classifications = classifier( self.task.test_vectors,\
        sequence=self.task.test_sequence, indices=self.task.test_indices, **test_add_args)

    return Result.from_task(self.task, classifications, dict(classifier.metadata))

def get_result(fold):
  """
  Needed for parallelized implementation, as we need a top-level function to pickle.
  """
  return fold.result

# TODO: Refactor in a way that allows access to per-fold classifiers
class Experiment(TaskSetResult):
  def __init__(self, taskset, learner=None, parallel=None):
    # TODO: Why is learner optional?
    self.taskset = taskset
    self.learner = learner
    self._results = None
    self.parallel = parallel if parallel is not None else config.getboolean('parameters', 'parallel_classify')

  @property
  def metadata(self):
    """ Result object metadata """
    result_metadata = dict(self.taskset.metadata)
    result_metadata['learner'] = self.learner.__name__
    result_metadata['learner_params'] = self.learner.params
    return result_metadata 
    
  @property
  def results(self):
    if self._results is None:
      self.run()
    return self._results

  @property
  def folds(self):
    folds = []
    for task in self.taskset:
      folds.append(ExperimentFold(task, self.learner))
    return folds

  def run(self, add_args = None):
    # TODO: parallelize over folds?
    results = []
    # TODO: Nicer in-progress output
    print "Experiment: %s %s" % (self.learner.__name__, self.taskset.metadata)
    try:
      if not self.parallel or not self.learner.is_pickleable():
        # TODO: Should we define a custom exception for this?
        raise cPickle.UnpickleableError
      cPickle.dumps(self.learner)
      # TODO: closing a multiprocessing pool produces an unsightly error msg
      # TODO: it seems that explicitly closing it does not cause this error msg
      pool = mp.Pool(config.getint('parameters','job_count'))
      for result in ProgressIter(pool.imap_unordered(get_result, self.folds), 'PARALLEL', maxval=len(self.taskset)):
        results.append(result)
      pool.close()
      pool.join() # This waits for all the pool members to exit

      results.sort(key=lambda x:x.metadata['index'])
    except (cPickle.UnpickleableError, TypeError):
      for fold in ProgressIter(self.folds, 'SERIES'):
        results.append(fold.result)
    self._results = results
    return results

