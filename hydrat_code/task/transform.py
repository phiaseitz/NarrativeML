from hydrat.task.taskset import TaskSet
from hydrat.task.task import Task
import time

# TODO: Merge this into the transformer interface???
def transform_task(task, transformer, add_args=None):
  if add_args is None:
    add_args = {}
  assert set(transformer.weights) == set(task.weights)
  transformer.weights = task.weights
  add_args['sequence']  = task.train_sequence
  add_args['indices'] = task.train_indices

  # TODO: create a timing context manager?
  start = time.time()
  transformer._learn(task.train_vectors, task.train_classes, add_args)
  learn_time = time.time() - start

  t = Task()

  start = time.time()
  for slot in Task.__slots__:
    if slot.endswith('vectors'):
      if slot.startswith('train'):
        add_args['sequence'] = task.train_sequence
        add_args['indices'] = task.train_indices
      elif slot.startswith('test'):
        add_args['sequence'] = task.test_sequence
        add_args['indices'] = task.test_indices
      setattr(t, slot, transformer._apply(getattr(task, slot), add_args))
    else:
      setattr(t, slot, getattr(task, slot))
  apply_time = time.time() - start

  # Separately update metadata
  t.metadata      = dict(task.metadata)
  t.metadata['feature_desc']+=(transformer.__name__,)
  if 'transform_learn_time' not in t.metadata:
    t.metadata['transform_learn_time'] = {}
  if 'transform_apply_time' not in t.metadata:
    t.metadata['transform_apply_time'] = {}
  t.metadata['transform_learn_time'][transformer.__name__] = learn_time
  t.metadata['transform_apply_time'][transformer.__name__] = apply_time
  return t
  
def transform_taskset(taskset, transformer, add_args=None):
  # TODO: How to handle add_args in terms of metadata??
  metadata = update_metadata(taskset.metadata, transformer)
  tasklist = [ transform_task(t, transformer, add_args) for t in taskset.tasks ]
  return TaskSet(tasklist, metadata)

def update_metadata(metadata, transformer):
  metadata = dict(metadata)
  # Eliminate feature name
  if 'feature_name' in metadata: 
    raise ValueError, "Should not be encountering feature_name"
  metadata['feature_desc']+=(transformer.__name__,)
  return metadata



