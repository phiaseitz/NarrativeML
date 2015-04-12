import pylab
import numpy

class ConfidenceIntervalPlot(object):

  def __init__( self
              , title          = None
              , xlabel         = None
              , ylabel         = None
              , barheight      = 0.4
              , barspace       = 0.1
              , bars_per_inch  = 5 
              ):
    self.barheight      = barheight 
    self.barspace       = barspace 
    self.title          = title
    self.xlabel         = xlabel
    self.ylabel         = ylabel
    self.bars_per_inch  = bars_per_inch
    self.__reset__()

  def __reset__(self):
    self.numbars = 0
    self.intervals = []
    self.interval_labels = []

  def add_interval(self, bounds, label):
    self.intervals.append(bounds)
    self.interval_labels.append(label)
    
  def _add_bar(self, lbound, ubound):
    pylab.broken_barh( [(lbound, ubound-lbound)]
                     , ( self.numbars * (self.barheight + self.barspace)
                       , self.barheight
                       )
                     )
    self.numbars += 1

  def render(self, path):
    barblockheight = self.barheight + self.barspace

    pylab.clf()
    #pylab.figure(figsize=(5,6))
    pylab.figure(figsize = (8, len(self.intervals)/ self.bars_per_inch))
    pylab.subplots_adjust(left = 0.25, right=0.95, bottom=0.05)
    for lbound, ubound in self.intervals:
      self._add_bar(lbound, ubound)
    pylab.yticks( numpy.arange( self.barheight/2.0
                              , self.numbars * barblockheight
                              , barblockheight 
                              )
                , self.interval_labels
                , size = 6 
                )
    pylab.ylim(0 - self.barspace, self.numbars * (self.barheight+self.barspace))
    if self.title:   pylab.title(self.title)
    if self.xlabel:  pylab.xlabel(self.xlabel)
    if self.ylabel:  pylab.ylabel(self.ylabel)
    pylab.savefig(path)


    
def test():
  plot = ConfidenceIntervalPlot()
  for i in xrange(50):
    plot.add_interval((0.96, 0.98), "a")
    plot.add_interval((0.94, 0.97), "b")
    plot.add_interval((0.92, 0.99), "c")
  plot.render('test.png')

if __name__ == "__main__":
  test()
