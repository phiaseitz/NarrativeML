import pylab
from matplotlib.font_manager import FontProperties

def bargraph( path
            , data
            , xlabel = "xlabel not set" 
            , ylabel = "ylabel not set" 
            , title  = "title not set" 
            , colors = ['blue', 'red', 'green']
            , data_ticks = None
            , bar_height = 1
            ):
  """
  Produce a bar graph.
  Input is the path to save the graph in, and some data to graph.
  Data format is a list of dictionaries. Each dictionary produces
  one subsection of the graph.
  """

  pylab.clf()
  base = 0
  if data_ticks and len(data_ticks) != len(data):
    raise ValueError, "data_ticks does not match number of data"

  ticks = []
  for data_index, subdata in enumerate(data):

    if data_ticks:
      #Compute where to put the tick along the axis
      tick_position = base + (float(len(subdata) * bar_height) - 1) / 2.0
      ticks.append((tick_position, data_ticks[data_index]))

    for i, (key, value) in enumerate(subdata.iteritems()):
      pylab.barh( base
                , value
                , height = bar_height
                , color = colors[ i % len(colors) ]
                , linewidth = 0
                , align = 'center'
                )
      pylab.text( value
                , base
                , key
                , verticalalignment = 'center'
                , horizontalalignment = 'right'
                , color='white'
                , fontproperties = FontProperties(size=4)
                )
      pylab.text( value
                , base
                , "%.2f" % (value) 
                , verticalalignment = 'center'
                , horizontalalignment = 'left'
                , color='black'
                , fontproperties = FontProperties(size=4)
                )
      base += bar_height 
    base += bar_height

  if data_ticks:
    tick_positions, tick_labels = zip(*ticks)
    #TODO: Dynamically size fonts
    pylab.yticks( tick_positions
                , tick_labels
                , fontproperties = FontProperties(size=4)
                )
  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)
  pylab.title(title)
  pylab.savefig(path, dpi=300)
  pylab.close()
  pylab.clf()


if __name__ == "__main__":
  print "Producing sample graphs"
  x = {'a': 1, 'b':2, 'c':3}
  y = {'a': 2, 'b':2}
  z = {'a': 2, 'b':1, 'd':4}

  bargraph('bargraph.png', [x,y,z], data_ticks=['x','y','z'])
