from networkx import DiGraph

class ClassNode(object): 
  count = 0
  def __init__(self, label):
    self.index = ClassNode.count
    ClassNode.count += 1
    self.label = label

  def __repr__(self):
    return str(self.index)

class ClassBranch(ClassNode): 
  def __init__(self, label):
    ClassNode.__init__(self, label)
    self.leaves = set()

class ClassLeaf(ClassNode): 
  def __init__(self, label):
    ClassNode.__init__(self, label) 

class ClassTree(DiGraph):
  def __init__(self):
    DiGraph.__init__(self)
    self.root = ClassBranch('Root') 
    self.branches = set([self.root])
    self.leaves = set()

