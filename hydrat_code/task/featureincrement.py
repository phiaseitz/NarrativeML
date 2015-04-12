"""
TaskSet based on incremental feature selection.

Marco Lui, July 2012
"""

from hydrat.datamodel import TaskSet
from hydrat.transformer.featureselect import FeatureSelect, KeepRule, HighestN
from hydrat.transformer.transform import transform_task

class FeatureIncrement(TaskSet):
  def __init__(self, taskset, weighting_function, keep_rules):
    """
    @param taskset A taskset to generate tasks from by feature selection
    @param featureselect A FeatureSelect instance
    @param counts A list of feature counts to generate tasks for
    """
    if len(taskset) != 1:
      raise ValueError("can only use length-1 tasksets")
    self.task = taskset[0]
    self.wf = weighting_function
    self.keep_rules = []
    self.__metadata = taskset.metadata
    for k in keep_rules:
      if isinstance(k, KeepRule):
        self.keep_rules.append(k)
      elif isinstance(k, int):
        self.keep_rules.append(HighestN(k))
      else:
        raise ValueError("{0} is not a KeepRule".format(k))

  @property
  def metadata(self):
    md = dict(self.__metadata)
    md['feature_desc'] += (self.wf.__name__,)
    md['keep_rules'] = map(str, self.keep_rules)
    return md

  def __len__(self):
    return len(self.keep_rules)

  def __getitem__(self, key):
    fs = FeatureSelect(self.wf, self.keep_rules[key])
    task = transform_task(self.task, fs)
    task.metadata['transform'] = fs.__name__
    return task
