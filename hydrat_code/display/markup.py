from contextlib import closing
import hydrat.common.markup as markup

oneliner = markup.oneliner

CSS_URL = "http://hum.cs.mu.oz.au/~mlui/lib/blue_style/style.css"
SCRIPTS = [ 
  "http://hum.cs.mu.oz.au/~mlui/lib/jquery-1.3.2.min.js",
  "http://hum.cs.mu.oz.au/~mlui/lib/jquery.metadata.min.js",
  "http://hum.cs.mu.oz.au/~mlui/lib/jquery-tablesorter-filter/tablesorter.js",
  "http://hum.cs.mu.oz.au/~mlui/lib/jquery-tablesorter-filter/tablesorter_filter.js",
  ]

class page(markup.page):
  """
  Represents a single HTML page
  """
  tablecount = 0
  sorter_code = []

  def __str__( self ):
    """
    We override str as we need to inject the tablesort code before rendering
    TODO: Handle multiple calls to __str__
    """
    self.css(CSS_URL)
    for s in SCRIPTS:
      self.script('', src=s, type='text/javascript')
    
    # build the final sorter script
    script = []
    script.append("$(document).ready( function() {")
    script.extend(self.sorter_code)
    script.append("});")
    self.script('\n'.join(script), type='text/javascript')
    return markup.page.__str__(self)

  def dict_table( self, rows, cols, col_headings=None, row_headings=None, 
      footer=None, title=None):
    if col_headings and len(col_headings) != len(cols):
        raise ValueError, "mismatch between cols and col_headings"
    if row_headings and len(row_headings) != len(rows):
        raise ValueError, "mismatch between rows and row_headings"

    # Assign new table identifier
    table_id = 'table%03d' % self.tablecount
    self.tablecount += 1


    # Implements searchboxes
    search_cols = []
    _col_headings = []
    if col_headings is not None:
      with closing(self.form.open(onsubmit="return false;")):
        with closing(self.table.open(style="width:auto")):
          for i,h in enumerate(col_headings):
            try:
              searchable = h['searchable']
            except (KeyError, TypeError):
              searchable = False 
            if searchable:
              with self.tr:
                col_id = '%s-%s' % ( table_id, filter(str.isalpha,h['label']) )
                with self.td: self.add(h['label'])
                with self.td: self.input(type='text',id=col_id+'-box')
                with self.td: self.input(type='submit',id=col_id+'-clear',value='Clear')
                search_cols.append('{ filterContainer: $("#%s-box")' % col_id )
                search_cols.append('filterClearContainer: $("#%s-clear")' % col_id )
                search_cols.append('filterColumns: [%d]}' % i )

    # write the script for searchboxes
    script = list()
    script.append('$("#%s").tablesorter({ widgets: ["zebra"] })'%table_id)
    script.append('.tablesorterFilter(' + ','.join(search_cols) + ');') 
    self.sorter_code.append('\n'.join(script))
    
    with closing(self.table.open(**{'class':'tablesorter', 'border':'0', 'id':table_id})):
      # write title
      if title: 
        self.caption(title)

      # pad headings if needed
      if row_headings is not None and col_headings is not None:
        col_headings = col_headings[:]
        col_headings.insert(0,'')

      # write headings
      if col_headings is not None:
        with self.thead:
          with self.tr:
            for c in col_headings:
              # handle structured headers that define the sort type
              if isinstance(c, basestring):
                self.th(c)
              else:
                if 'sorter' in c:
                  sorter = "'%s'"% c['sorter'] if c['sorter'] else "false"
                  self.th(c['label'], **{'class':"{sorter: %s}" % sorter})
                else:
                  self.th(c['label'])

      with self.tbody:
        for i, row in enumerate(rows):
          with self.tr:
            if row_headings is not None:
              self.th(row_headings[i])
            for c in cols:
              self.td(str(rows[i][c]) if c in rows[i] else 'UNKNOWN')

      if footer is not None:
        with self.tfoot:
          with self.tr:
            if row_headings is not None:
              self.th()
            for c in cols:
              self.th(footer[c])
        

if __name__ == "__main__":
  table_size = 10
  lines = [ dict( (j,j*i) for j in xrange(table_size)) for i in xrange(table_size)]
  headings = [ 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven'
             , 'eight', 'nine' ][:table_size]
  col_headings = [ dict(label=h,searchable=True,sorter='digit') for h in headings ]

  p=page()
  p.init()
  p.dict_table(lines, xrange(table_size), col_headings=col_headings, row_headings=headings, title='Multiplication')
  p.dict_table(lines, xrange(table_size), col_headings=col_headings, row_headings=headings, title='Multiplication')
  print p()



        


    
