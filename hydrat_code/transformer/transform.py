from hydrat.datamodel import TaskSet, BasicTask
def transform_task(task, transformer):
    train_sequence = task.train_sequence
    train_indices = task.train_indices
    train_vectors = task.train_vectors

    # Transform each task
    add_args = {}
    add_args['sequence'] = train_sequence
    add_args['indices'] = train_indices

    # Patch transformer with our known weights
    # TODO: Why can't weights just be an argument?
    weights = transformer.weights
    transformer.weights = task.weights

    transformer._learn(train_vectors, task.train_classes, add_args)
    train_vectors = transformer._apply(train_vectors, add_args)

    # Transform test vectors
    add_args = {}
    add_args['sequence'] = task.test_sequence
    add_args['indices'] = task.test_indices
    test_vectors = transformer._apply(task.test_vectors, add_args)


    t = BasicTask(
      train_vectors, 
      task.train_classes, 
      task.train_indices,
      test_vectors, 
      task.test_classes, 
      task.test_indices,
      task.train_sequence, 
      task.test_sequence,
      metadata= dict(task.metadata)
      )

    # Copy weights back into task
    transformer.weights = weights
    return t

class Transform(TaskSet):
  def __init__(self, taskset, transformer):
    self.taskset = taskset 
    self.transformer = transformer

  """
  # TODO: figure out why this is here in the first place.
  def __getattr__(self, key):
    if key in self.__dict__:
      return self.__dict__[key]
    else:
      return getattr(self.taskset, key)
  """

  @property
  def metadata(self):
    # TODO: how about parametrized transformers?
    metadata = dict(self.taskset.metadata)
    metadata['feature_desc'] += (self.transformer.__name__,)
    metadata['variant'] = None
    return metadata

  def __len__(self):
    return len(self.taskset)

  def __getitem__(self, key):
    # TODO: Work out why we needed add_args, and what to do with it now
    # TODO: Timing of the component parts
    task = self.taskset[key]
    retval = transform_task(task, self.transformer)
    return retval

