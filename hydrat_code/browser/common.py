import hydrat.common.markup as markup

#TODO: Serve jquery & css from a location packaged with the module.
#TODO: Allow for customization of the summary function and/or display headers
CSS_URL = "http://hum.cs.mu.oz.au/~mlui/lib/blue_style/style.css"
def navbar():
  page = markup.page()
  with page.table:
    with page.tr:
      with page.td: page.a('spaces',    href='/spaces')
      with page.td: page.a('datasets',  href='/datasets')
      with page.td: page.a('tasks',     href='/tasks')
      with page.td: page.a('results',   href='/results')
  return str(page)

page_config=\
  { 'css'      : CSS_URL
  , 'charset'  : 'utf8'
  , 'header'   : navbar()
  }


