import cherrypy
import urllib
import StringIO
import hydrat.display.markup as markup
from common import page_config
from display import list_as_html, dict_as_html, list_of_links

class Spaces(object):
  def __init__(self, store, bconfig):
    self.store = store
    self.bconfig = bconfig

  @cherrypy.expose
  def index(self):
    page = markup.page()
    page.init(**page_config)

    cols = ['name', 'size', 'encoding']
    headings = ['Name', {'label':'Size', 'sorter':'digit'}, 'Encoding']

    page.h1('Feature Spaces')
    rows = [self.store.get_SpaceMetadata(s) for s in self.store.list_FeatureSpaces()]
    for r in rows:
      r['name'] = markup.oneliner.a(r['name'], href='view?' +urllib.urlencode({'name':r['name']}))
    page.dict_table( rows, cols, col_headings=headings)

    page.h1('Class Spaces')
    rows = [self.store.get_SpaceMetadata(s) for s in self.store.list_ClassSpaces()]
    for r in rows:
      r['name'] = markup.oneliner.a(r['name'], href='view?' +urllib.urlencode({'name':r['name']}))

    page.dict_table( rows, cols, col_headings=headings)
    return str(page)

  @cherrypy.expose
  def view(self, name):
    encoding = self.store.get_SpaceMetadata(name)['encoding']
    page = markup.page()
    page.init(**page_config)
    with page.ul:
      for item in self.store.get_Space(name):
        with page.li:
          if isinstance(item, unicode):
            page.add(str(item.encode(encoding)))
          else:
            page.add(str(item))
    return str(page)


