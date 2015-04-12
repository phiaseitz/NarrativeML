raise DeprecationWarning, "Needs to be fixed up"
from common import proportion_allocate
from common.debugtools import debugInfo
from trainandtest import trainandtest 
from common import flatten

class incremental:
  """
  Incrememntal testing class
  Produces a number of tasks, representing a scaling of data.
  Needs to allow for scaling of training data and of test data.
  """

  def __init__( self, train_proportions, test_proportions
              , dataset, model
              , fs = None, fsnum = None ):
    debugInfo("Setting up incremenal testing on %s"%(dataset.__name__))
    self.dataset = dataset
    self.model = model

    self.fs = fs
    self.fsnum = fsnum

    train_ids, test_ids = proportion_allocate( dataset.docids,
                                               [sum(train_proportions), sum(test_proportions)],
                                               randomise = True
                                             )
    self.train_parts = proportion_allocate( train_ids,
                                            train_proportions,
                                          )
    self.test_parts  = proportion_allocate( test_ids,
                                            test_proportions,
                                          )
    self.desc = "incremental tasks for %s" % (self.dataset.__name__)

  def getTasks(self):
    def concat(list): return reduce(lambda x, y:x+y, list)
    for i in xrange(len(self.train_parts)):
      train_ids = concat(self.train_parts[:i+1])
      for j in xrange(len(self.test_parts)):
        test_ids = concat(self.test_parts[:j+1])
        tt = trainandtest( self.dataset
                         , self.model
                         , train_ids = train_ids
                         , test_ids = test_ids
                         , label = "tr%d_te%d"%(len(train_ids), len(test_ids))
                         )

        if self.fs is not None and self.fsnum is not None:
          tt.featureselect(self.fs, self.fsnum)
          self.fslabel = tt.fslabel
          self.fsname  = tt.fsname
          self.fsnum   = tt.fsnum
        else:
          self.fslabel = "nofs"
          self.fsname  = None
          self.fsnum   = None
        this_task = tt.getTask()
        this_task.annotations.update({ "tr_size" : len(train_ids)
                                     , "te_size" : len(test_ids)
                                    })
        yield this_task
