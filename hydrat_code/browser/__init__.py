"""
Hydrat interactive store browser based on CherryPy
"""
import cherrypy
import urllib
import StringIO
import hydrat.common.markup as markup

from common import page_config
from spaces import Spaces
from dataset import Datasets
from task import Tasks
from result import Results
from display import list_of_links



class StoreBrowser(object):
  def __init__(self, store, bconfig):
    self.store = store
    self.results = Results(store, bconfig)
    self.datasets = Datasets(store)
    self.spaces = Spaces(store, bconfig)
    self.tasks = Tasks(store, bconfig)

  @cherrypy.expose
  def index(self):
    links =\
      [ ( 'Spaces', 'spaces')
      , ( 'Datasets', 'datasets')
      , ( 'Tasks', 'tasks')
      , ( 'Results', 'results')
      ]
    page = markup.page()
    page.init(**page_config)
    page.add(list_of_links(links))
    return str(page)
