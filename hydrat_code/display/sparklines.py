import Image, ImageDraw
import StringIO
import urllib
import numpy
from scipy.stats import scoreatpercentile

# Based on http://bitworking.org/news/Sparklines_in_data_URIs_in_Python
def divergence_bar(results, baseline = None, height = 20, width = 10, gap = 2):
  # Use the mean as a baseline if none is provided.
  baseline = baseline if baseline is not None else float(sum(results)) / len(results)
  im = Image.new("RGBA", (len(results)*(width + gap), height), (0,0,0,0))
  draw = ImageDraw.Draw(im)
  limit = max(map(lambda x : abs( x - baseline ), results))
  scaling_factor = height / 2.0 / limit
  mid = int(round(height/2.0))
  for (r, i) in zip(results, range(0, len(results)*(width+gap), (width+gap))):
    diff = abs(r - baseline)
    prop_diff = diff * scaling_factor
    if r >= baseline:
      draw.rectangle((i, mid, i+width-1, mid - prop_diff ), fill='blue')
    else:
      draw.rectangle((i, mid, i+width-1, mid + prop_diff ), fill='red')
  del draw

  f = StringIO.StringIO()
  im.save(f, "PNG")
  return 'data:image/png,' + urllib.quote(f.getvalue())

def barchart(values, height = 20, width = 5, gap = 1, color='black'):
  limit = float(max(values))
  im = Image.new("RGBA", (len(values)*(width + gap), height), (0,0,0,0))
  draw = ImageDraw.Draw(im)
  scaling_factor = height / limit

  for (r, i) in zip(values, range(0, len(values)*(width+gap), (width+gap))):
    draw.rectangle( (i, height , i + width - 1, height - (r*scaling_factor) ) ,fill=color)
  del draw

  f = StringIO.StringIO()
  im.save(f, "PNG")
  return 'data:image/png,' + urllib.quote(f.getvalue())

def histogram(values, bins = 100, height = 20, width = 1, gap = 1, color = 'black'):
  #hist,bounds = numpy.histogram(values, bins = bins, new=True)
  hist,bounds = numpy.histogram(values, bins = bins)
  return barchart(hist, height, width, gap, color)

def boxplot(values, width=300, height=20, range=None):
  """ Produce a box and whisker plot """
  if range is None:
    minval = float(min(values))
    maxval = float(max(values))
    gap = maxval - minval
    range = (minval - 0.1 * gap, maxval + 0.1 * gap)
  assert range[0] < range[1]
  rangemin = float(range[0])
  rangemax = float(range[1])
  gap = rangemax - rangemin

  # Compute the five-number-summary
  # based on: 
  #  http://stackoverflow.com/questions/3878245/tukey-five-number-summary-in-python
  #  Probability & Statistical Inference 7e Hogg & Tanis pp330
  q1 = scoreatpercentile(values,25)
  q3 = scoreatpercentile(values,75)
  md = numpy.median(values)
  av = numpy.mean(values)
  minval = min(values)
  maxval = max(values) 
  #iqd = q3-q1
  #inner_fence = 1.5*iqd
  #outer_fence = 3.0*iqd
  #print range, (minval, q1, md, q3, maxval)

  # Geometry calculations
  topgap = 3
  def project(v): return (v - rangemin) / gap * width
  boxl = project(q1)
  boxr = project(q3)
  whiskerl = project(minval)
  whiskerr = project(maxval)
  whisker = height / 2
  median = project(md)
  average = project(av)

  # Draw 
  im = Image.new("RGBA", (width, height), (0,0,0,0))
  draw = ImageDraw.Draw(im)

  # IQR Box
  draw.rectangle( ( boxl, topgap, boxr, height - topgap) , outline='black')
  # L whisker
  draw.line( [(whiskerl, whisker), (boxl, whisker)], fill='black' )
  # R whisket
  draw.line( [(boxr, whisker), (whiskerr, whisker)], fill='black' )
  # Median
  draw.line( [(median, topgap), (median, height-topgap)], fill='black' )
  # Average
  draw.line( [(average, topgap+2), (average, height-topgap-2)], fill='red' )
  del draw

  f = StringIO.StringIO()
  im.save(f, "PNG")
  return 'data:image/png,' + urllib.quote(f.getvalue())
