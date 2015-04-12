from cStringIO import StringIO
import numpy as n

def default_text_method(text):
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text
    elif isinstance(text, int):
      return ("%d" % text)
    elif isinstance(text, long):
      return ("%d" % text)
    elif isinstance(text, float):
      return ("%.3f" % text)
    elif isinstance(text, bool):
      return ("1" if text else "0")
    elif issubclass(type(text), n.integer):
      return str(text)
    elif isinstance(text, tuple):
      return str(text)
    elif isinstance(text, dict):
      return str(text)
    else:
      return ('&nbsp;')


class HTMLWriter(object):
  """
  SAXish API for writing HTML. No complicance checking, could really be considered 
  an XML writer except for the table-specific functionality.
  Was originally intended as an implementation of a unified display API, so we could 
  render HTML or Latex interchangably, but we have veered far off course on this.
  """
  def __init__( self, target, text_method = default_text_method):
    self.target = target
    self.text_method = text_method
    assert callable(text_method)

  def _newline(self):
    self.target.write('\n')

  def _text(self, text):
    self.target.write(self.text_method(text))

  def _openTag(self, tag, attrs = None):
    if attrs is not None:
      attr_str     = ' '.join( (a+'="'+attrs[a]+'"') for a in attrs)
      tag_contents = '%s %s' % (tag, attr_str) 
    else:
      tag_contents = tag
    self.target.write('<%s>' % tag_contents)

  def _closeTag(self, tag):
    self.target.write('</%s>\n' % tag)

  def _emptyTag(self, tag, attrs=None):
    if attrs is not None:
      attr_str     = ' '.join( (a+'="'+attrs[a]+'"') for a in attrs)
      tag_contents = '%s %s' % (tag, attr_str) 
    else:
      tag_contents = tag
    self.target.write('<%s />' % tag_contents)
    
  def _enclose(self, tag, text, attrs = None):
    # We don't auto-detect emptiness because some situations need an explicit empty pair
    # e.g. <script> for js inclusion
    self._openTag(tag, attrs)
    self._text(text)
    self._closeTag(tag)

  def line(self, text):
    self._text(text)
    self.linebreak()

  def linebreak(self):
    self.target.write('<br/>')

  def section(self, text):
    self._enclose('h2', text)

  def subsection(self, text):
    self._enclose('h4', text)

  def paragraph(self, text):
    self._openTag('p')
    for line in text.split('\n'):
      self._text(line)
      self.linebreak()
    self._closeTag('p')

  def image(self, src):
    self._openTag('img', {'src':src})

  def link(self, target, text):
    self._enclose('a', text, {'href':target})

  def _caption(self, caption):
    self._openTag('caption')
    self._text(caption)
    self._closeTag('caption')
    self._newline()

  def _cell(self, text, type):
    self._openTag(type)
    self._text(text)
    self._closeTag(type)
    
  def _row(self, cells, celltype = 'td', header = None):
    self._openTag('tr')
    if header is not None:
      self._cell(header, 'th')
    for cell in cells:
      self._cell(cell, celltype)
    self._closeTag('tr')
    self._newline()

  def _heading(self, cells):
    self._openTag('thead')
    # Add a space after heading labels, to allow for the sort icons.
    cells_str = [ self.text_method(c) + ' ' for c in cells ]
    self._row(cells, celltype = 'th')
    self._closeTag('thead')

  def _start_table( self
                  , col_headings
                  , row_headings
                  , title
                  , attrs_override = {}
                  ):
    attrs = { 'border':'1' }
    attrs.update(attrs_override)
    self._openTag('table', attrs)
    self._newline()
    if title: 
      self._caption(title)
    if row_headings is not None and col_headings is not None:
      # Add a blank if we are drawing row headings as well
      col_headings = col_headings[:]
      col_headings.insert(0,'')
    if col_headings is not None:
      self._heading(col_headings)
    self._openTag('tbody')

  def _footing(self, cells):
    self._openTag('tfoot')
    self._row(cells, celltype = 'td')
    self._closeTag('tfoot')

  def _end_table(self, footing = None, row_headings = None):
    self._closeTag('tbody')
    if footing is not None:
      if row_headings is not None:
        footing.insert(0,'')
      self._footing(footing)
    self._closeTag('table')
    self._newline()

  def _dict_table_rows( self, rows, cols, row_headings ):
    for row_index in xrange(len(rows)):
      if row_headings is not None:
        heading = row_headings[row_index] 
      else:
        heading = None

      entries = []
      for c in cols:
        if c in rows[row_index]:
          entries.append(rows[row_index][c])
        else:
          entries.append("UNKNOWN")
      self._row(entries, header=heading)

  def dict_table( self
                , rows 
                , cols
                , col_headings  = None
                , row_headings  = None
                , title         = None
                ):
    if col_headings: 
      assert len(col_headings) == len(cols)
      if row_headings:
        assert len(row_headings) == len(rows)
    self._start_table( col_headings, row_headings, title )
    self._dict_table_rows( rows, cols, row_headings )
    self._end_table()

  def _array_table_rows( self, arr, row_headings ):
    for row_index in xrange(len(arr)):
      if row_headings is not None:
        heading = row_headings[row_index] 
      else:
        heading = None
      self._row(list(arr[row_index]), header=heading)

  def array_table( self
                 , arr 
                 , col_headings  = None
                 , row_headings  = None
                 , title         = None
                 , col_footings  = None
                 ):
    assert len(arr.shape) == 2
    rows, cols = arr.shape
    if col_headings is not None:
      assert len(col_headings) == cols
    if row_headings is not None:
      assert len(row_headings) == rows
    self._start_table( col_headings, row_headings, title )
    self._array_table_rows( arr, row_headings )
    self._end_table( col_footings, row_headings )

class TableSort(HTMLWriter):
  """
  Writer with sortable tables using jquery
  """
  js_urls =  [ "http://hum.cs.mu.oz.au/~mlui/lib/jquery-1.3.2.min.js"
             , "http://hum.cs.mu.oz.au/~mlui/lib/jquery.metadata.min.js"
             , "http://hum.cs.mu.oz.au/~mlui/lib/jquery-tablesorter-filter/tablesorter.js"
             , "http://hum.cs.mu.oz.au/~mlui/lib/jquery-tablesorter-filter/tablesorter_filter.js"
             ]
  style_url       = "http://hum.cs.mu.oz.au/~mlui/lib/blue_style/style.css"

  jquery_script = \
"""
  $("table").tablesorter({ widgets: ['zebra'] });
"""

  def __init__( self, *args, **kwargs):
    HTMLWriter.__init__( self, *args, **kwargs)
    self.tablecount = 0

  def __enter__(self):
    self._text(r'''<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
      "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">''')
    self._openTag('html', { 'xmlns':'http://www.w3.org/1999/xhtml'
                          , 'xml:lang':'en-us'
                          }
                 )
    self.final_target = self.target

    # Set up in memory targets to build document components in
    self.head_target = StringIO()
    self.body_target = StringIO()

    self.target = self.head_target
    self._openTag('head')
    # Add in the CSS link
    self._emptyTag('link', { 'rel'   : 'stylesheet'
                                       , 'href'  : self.style_url
                                       , 'type'  : 'text/css'
                                       , 'media' : 'print, projection, screen'
                                       }
                              )

    # Add in libraries to our head
    # For some reason we can't just do this with an empty tag.
    for url in self.js_urls:
      self._enclose( 'script'
                  , ''
                  , { 'type' : 'text/javascript'
                    , 'src'  : url
                    }
                  )
    # Open a script tag so we can write in bits of script as we
    # write new tables
    self._openTag( 'script', { 'type' : 'text/javascript' } )
    self._text("$(document).ready( function() {\n")

    # Switch to writing to body
    self.target = self.body_target
    self._openTag('body')
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_type is not None: return False
    # Seal off the body
    self._closeTag('body')

    # Tidy up the head
    self.target = self.head_target

    self._text("});\n")
    self._closeTag('script')
    self._closeTag('head')

    # Finish writing to final target
    self.target = self.final_target
    self.target.write(self.head_target.getvalue())
    self.target.write(self.body_target.getvalue())
    self._closeTag('html')

    return False

  def _start_table( self
                  , col_headings
                  , row_headings
                  , title
                  , attrs = {'class':'tablesorter', 'border':'0'}
                  ):
    # Overload to remove border writing but set sortable table 
    table_id = 'table%03d' % self.tablecount
    self.tablecount += 1

    # Write javascript that enables table sorting
    self.head_target.write('$("#%s")\n'%table_id)
    self.head_target.write('  .tablesorter({ widgets: ["zebra"] })\n')

    search_cols = []
    if col_headings is not None:
      for i,h in enumerate(col_headings):
        try:
          searchable = h['searchable']
        except KeyError:
          searchable = False 
        except TypeError:
          searchable = False 
        if searchable:
          col_id = '%s-%s' % ( table_id, filter(str.isalpha,h['label']) )
          self._text(h['label']+": ")
          self._emptyTag('input', {'type':'text','id':col_id+'-box'})
          self._emptyTag('input', {'type':'submit','id':col_id+'-clear','value':'Clear'})
          self._emptyTag('br')
          self._text('\n')
          search_cols.append((i, col_id))
    # Write javascript that sets up filtering
    self.head_target.write('  .tablesorterFilter(\n')
    self.head_target.write(','.join(   '{ filterContainer: $("#%s-box")' % col_id \
                                      +', filterClearContainer: $("#%s-clear")' % col_id \
                                      +', filterColumns: [%d]}' % i \
                                   for  i, col_id 
                                   in   search_cols
                                   )
                          )
    self.head_target.write(');\n')

    attrs['id'] = table_id
    HTMLWriter._start_table(self, col_headings, row_headings, title, attrs)

  def _cell(self, text, celltype):
    # Overload to force correct type of parsing for sort in headers
    if celltype == 'th':
      if type(text) == dict:
        contents = text['label']
        # Use None to specify unsortable
        if 'sorter' in text:
          sorter = "'%s'"% text['sorter'] if text['sorter'] else "false"
          self._enclose(celltype, contents, {'class':"{sorter: %s }" % sorter})
        else:
          self._enclose(celltype, contents)
      else:
        self._enclose(celltype, text)
    else:
      self._enclose(celltype, text)
    

def table_str( array, col_headings = None, row_headings = None, blank_on_zero=False):
  output = StringIO()
  writer = HTMLWriter(output)
  if blank_on_zero:
    def custom_cell(text, type):
      if text == 0.0:
        text = ''
      if isinstance(text, float):
        text = int(text)
      else:
        text = str(text)
      writer._openTag(type)
      writer._text(text)
      writer._closeTag(type)
    writer._cell = custom_cell
  writer.array_table( array
                    , col_headings = col_headings
                    , row_headings = row_headings
                    )
  return output.getvalue()

  
if __name__ == '__main__':
  import sys

  table_size = 10
  lines = [ dict( (j,j*i) for j in xrange(table_size)) for i in xrange(table_size)]
  headings = [ 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven'
             , 'eight', 'nine' ]

  renderer = HTMLWriter(sys.stdout)
  renderer.dict_table( lines
                     , xrange(table_size)
                     , col_headings = headings
                     , row_headings = headings
                     , title= 'Multiplication Table'
                     )

