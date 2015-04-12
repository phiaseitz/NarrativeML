from hydrat.classifier.abstract import Learner, Classifier
from hydrat.result.interpreter import SingleHighestValue
from hydrat.result.result import Result
from networkx import DiGraph
from ClassTree import ClassBranch
import numpy

class ClassifierNode(object):
  def __init__(self, id, leaves, successor_ids, classifier):
    self.classifier = classifier
    self.id = id
    self.leaves = leaves
    self.successor_ids = successor_ids
    self.indices = []

class ClassifierLeaf(object):
  def __init__(self, id, label):
    self.id = id
    self.label = label 
    self.indices = []

class ClassTreeClassifier(Classifier):
  __name__ = 'classtreeclassifier'
  def __init__(self, class_tree, root, classlabels, interpreter = None, analysis = False):
    Classifier.__init__(self)
    self.tree = class_tree
    self.root = root
    self.classlabels = classlabels
    if interpreter is None:
      self.interpreter = SingleHighestValue()
    else:
      self.interpreter = interpreter
    self.analysis = analysis

  def _classify(self, feature_map):
    num_instances = len(feature_map)
    num_classes = len(self.classlabels) 
    indices = numpy.arange(num_instances)
    class_indexes = dict( (l, i) for i,l in enumerate(self.classlabels))
    class_map = numpy.zeros((num_instances,num_classes), dtype='bool')

    # Each queue member is a 3-tuple:
    #  1) node of the class tree being operated on
    #  2) feature map subset relevant to the node
    #  3) indices of the subset in the main feature map
    queue = [(self.root, feature_map, indices)]
    while len(queue) > 0:
      n, n_instances, n_indices = queue.pop(0)
      self.logger.debug("classifying %d instances", len(n_instances) )
      classifications = n.classifier(n_instances)

      if self.analysis:
        n.classifications = classifications
        n.indices = n_indices

      if classifications.dtype == bool:
        self.logger.warning('Boolean Classification Output')
      else:
        self.logger.debug('Interpreting result using %s', str(self.interpreter))
        classifications = self.interpreter(classifications)

      for s in self.tree.successors(n):
        i = n.successor_ids.index(s.id)
        s_mask = classifications[:,i]
        if s_mask.sum() == 0:
          continue
        s_instances = n_instances[s_mask]
        s_indices   = n_indices[s_mask]
        if isinstance(s, ClassifierLeaf):
          for index in s_indices:
            class_map[index, class_indexes[s.label]] = True 
          if self.analysis:
            s.indices = s_indices
        else:
          queue.append((s, s_instances, s_indices)) 

    return class_map

  def analyse(self, goldstandard):
    """
    Provides a view on the inner workings of the tree classifier.
    """
    if not self.analysis:
      raise ValueError, "ClassTreeClassifier analysis was not enabled."

    class_indexes = dict( (l, i) for i,l in enumerate(self.classlabels))
    queue = [self.root]
    results = []
    while len(queue) > 0:
      n = queue.pop(0)
      cl_indices = n.indices

      if isinstance(n, ClassifierLeaf): 
        gs_class_indices = [class_indexes[n.label]]
      else:
        gs_class_indices = sorted(class_indexes[l] for l in n.leaves)

      gs_instance_mask = goldstandard[:,gs_class_indices].sum(axis=1)
      gs_indices = gs_instance_mask.nonzero()[0] 
      results.append((n.id, set(gs_indices), set(cl_indices)))
      successors = self.tree.successors(n)
      queue.extend(successors)
    return results

class ClassTreeLearner(Learner):
  __name__ = 'classtreelearner'
  def __init__(self, tree, classlabels, learner):
    Learner.__init__(self)
    self.tree = tree
    self.classlabels = classlabels 
    self.learner = learner
    self.__name__ = 'classtree-' + learner.__name__

  def _learn(self, feature_map, class_map):
    classifier_tree = DiGraph()
    root = self.train_classifier_node(self.tree.root, feature_map, class_map)

    def grow_tree(struct_root, classif_root):
      for s in self.tree.successors(struct_root):
        node = self.train_classifier_node(s, feature_map, class_map)
        classifier_tree.add_node(node)
        classifier_tree.add_edge(classif_root, node)
        grow_tree(s, node)

    # Recursively populate the tree with classifier nodes
    grow_tree(self.tree.root, root)
    
    return ClassTreeClassifier(classifier_tree, root, self.classlabels)

  def train_classifier_node(self, root, feature_map, class_map):
    successors = self.tree.successors(root)
    if successors == []:
      return ClassifierLeaf(root.index, root.label)

    class_indexes = dict( (l, i) for i,l in enumerate(self.classlabels))
    num_instances = len(feature_map)
    num_successors = len(successors)
    node_classes = root.leaves

    node_lang_indices = sorted(class_indexes[l] for l in node_classes)
    node_instance_mask = class_map[:,node_lang_indices].sum(axis=1, dtype='bool')
    node_instance_indices = numpy.arange(num_instances)[node_instance_mask]
    self.logger.debug("Learning a classification tree node")
    self.logger.debug("%d instances, %d classes in %d branches", len(node_instance_indices),len(node_lang_indices), num_successors)

    initial_class_map = numpy.empty((num_instances, num_successors) ,dtype='bool')
    successor_ids = []

    for i,s in enumerate(successors):
      if isinstance(s, ClassBranch):
        class_indices = [class_indexes[l] for l in s.leaves]
        initial_class_map[:,i] = class_map[:,class_indices].sum(axis = 1)
      else:
        class_index = class_indexes[s.label]
        initial_class_map[:,i] = class_map[:,class_index]
      successor_ids.append(s.index)
        
    node_feature_map = feature_map[node_instance_indices]
    node_class_map = initial_class_map[node_instance_indices]

    node_classifier = self.learner(node_feature_map, node_class_map)

    # We use the indexes to keep track of the tree structure, so
    # that we can map classifier output to the correct successor
    # nodes
    return ClassifierNode( root.index 
                         , root.leaves
                         , successor_ids
                         , node_classifier
                         )


