import cherrypy
import urllib
import numpy
import StringIO
from collections import defaultdict

import hydrat.display.markup as markup
from hydrat.display.html import TableSort
from hydrat.common import as_set
from hydrat.display.tsr import result_summary_table, project_compound
from hydrat.result import classification_matrix
from hydrat.common.metadata import metamap

import hydrat.result.stats as stats
from display import list_as_html, dict_as_html, list_of_links
from common import page_config
from hydrat.store import NoData


KEY_SEP =':'

def results_metadata_map(store, params, max_uniq = 10):
  mapping = metamap( t.metadata for t in store.get_TaskSetResults(params) )
  for key in mapping.keys():
    if len(mapping[key]) > max_uniq or len(mapping[key]) <= 1:
      del mapping[key]
  return mapping

from hydrat.summary import Summary
class Navigation(Summary):
  def init(self, result, interpreter):
    Summary.init(self, result, interpreter)
    self.uuid = str(result.uuid)

  def key__link(self):
    link = markup.oneliner.a('link', href='view?'+urllib.urlencode({'uuid':self.uuid}))
    return str(link)
    
  def key__pairs(self):
    link = markup.oneliner.a('link', href='confusion?'+urllib.urlencode({'uuid':self.uuid}))
    return str(link)

  def key__select(self):
    link = markup.oneliner.input(type='checkbox', name='uuid', value=self.uuid)
    return str(link)

  # TODO: Offer this as an option associated with 'select' instead.
  def key__delete(self):
    link = markup.oneliner.a('delete', href='delete?'+urllib.urlencode({'uuid':self.uuid}))
    return str(link)

class Results(object):
  def __init__(self, store, bconfig):
    self.store = store
    self.summary_fn = bconfig.summary_fn
    self.interpreter = bconfig.interpreter
    self.relevant = bconfig.relevant

  @cherrypy.expose
  def index(self):
    raise cherrypy.HTTPRedirect('list')

  def result_summary_page(self, params, page, ext_summary_fn = None, relevant = None):
    uuids = self.store._resolve_TaskSetResults(params)
    int_id = self.interpreter.__name__

    summary_fn = Navigation()
    if ext_summary_fn is not None:
      summary_fn.extend(ext_summary_fn)

    summaries = self.get_full_summary(uuids, summary_fn)

    page.h3('Interpreter')
    page.add(dict_as_html({'Interpreter':int_id}))

    page.h3('Parameters')
    page.add(dict_as_html(params))

    if len(summaries) == 0:
      page.h1('No results')
    else:
      text = StringIO.StringIO()

      # Show all our metadata if no filter is specified
      if relevant is None:
        #TODO: hydrat.display.tsr.result_summary_table does this by default, need to refactor against that.
        relevant = [(k.title(),k) for k in sorted(summaries[0].keys()) if not k.startswith('_')]
      else:
        relevant = relevant[:]

      relevant.insert(0, ({'label':'Select','sorter':None},'_select') )
      relevant.append( ( {'sorter': None, 'label':"Details"}      , "_link"          ) )
      relevant.append(("Pairs", '_pairs'))
      if self.store.mode == 'a':
        relevant.insert(0,("Delete", '_delete'))

      with TableSort(text) as renderer:
        result_summary_table(summaries, renderer, relevant)

      page.p('Displaying %d results' % len(uuids))

      page.form(action='receive', method='post')
      page.input(type='submit', name='action', value='csv')
      page.input(type='submit', name='action', value='compare')
      page.input(type='submit', name='action', value='metamap')
      if self.store.mode == 'a':
        page.input(type='submit', name='action', value='delete')
      page.br()
      page.add(text.getvalue())
      page.form.close()

  @cherrypy.expose
  def receive(self, **params):
    if 'action' in params:
      action = params['action']
      del params['action']
      raise cherrypy.HTTPRedirect(action+'?'+urllib.urlencode(params, True))
    else:
      raise cherrypy.HTTPRedirect("list")

  @cherrypy.expose
  def list(self, **params):
    page = markup.page()
    page.init(**page_config)

    # Show contstraint options
    mapping = results_metadata_map(self.store, params)
    param_links = {}
    for key, values in mapping.iteritems():
      links = []
      new_params = dict(params)
      for v in values:
        label = '%s(%d)' % (v, values[v])
        if isinstance(v, str):
          new_params[key] = v
          links.append( markup.oneliner.a(label, href='list?'+urllib.urlencode(new_params, True)))
        else:
          links.append( label )
      param_links[key] = links
      
    page.add(dict_as_html(param_links))

    # Link to detailed results for these parameters
    with page.p:
      #TODO: This is how we can parametrize which key-value pairs to show
      page.a('Show detailed results', href='details?'+urllib.urlencode(params))

    # Draw the actual summary
    self.result_summary_page(params, page)
    return str(page)

  @cherrypy.expose
  def details(self, **params):
    page = markup.page()
    page.init(**page_config)
    self.result_summary_page(params, page, self.summary_fn, self.relevant)
    return str(page)

  @cherrypy.expose
  def compare(self, uuid, show_wrong='0', goldstandard=None):
    print "show_wrong", show_wrong, type(show_wrong)
    # TODO: Parametrize interpreter for non one-of-m highest-best results
    # TODO: Add a count of # of compared result which are correct
    # TODO: show metadata keys in which the results differ
    from hydrat.common import as_set
    from hydrat.result.interpreter import SingleHighestValue
    interpreter = SingleHighestValue()
    uuid = as_set(uuid)

    # Hardcode interpreter
    interpreter = SingleHighestValue()

    # Read results
    results = [ self.store._get_TaskSetResult(i) for i in uuid ]
    md = results[0].metadata
    ds_key = 'eval_dataset' if 'eval_dataset' in md else 'dataset'

    # Sanity check
    must_match = ['class_space']
    for m in must_match:
      value_set = set(r.metadata[m] for r in results)
      if len(value_set) != 1:
        raise ValueError, "Non-uniform value for '%s' : %s" % (m, value_set)
    # TODO: Check that the instance IDs match

    # Grab relevant data from store
    class_space = self.store.get_Space(md['class_space'])
    instance_ids = self.store.get_InstanceIds(md[ds_key])
    gs = self.store.get_ClassMap(md[ds_key], md['class_space']).raw

    classifs = []
    for result in results:
      cl = result.overall_classification(range(len(instance_ids)))
      cl = cl.sum(axis=2)
      cl = interpreter(cl)
      classifs.append(cl)
    classifs = numpy.dstack(classifs)

    # Identify classes that neither GS nor CL utilize, we can just skip these.
    boring_cl_unused = numpy.logical_and(gs.sum(axis=0) == 0, classifs.sum(axis=2).sum(axis=0) == 0)
    int_cl = numpy.logical_not(boring_cl_unused)

    boring_inst_allright = (numpy.logical_and((classifs.sum(axis=2) == len(uuid)), gs).sum(axis=1) == 1)
    boring_inst_allwrong = (numpy.logical_and((classifs.sum(axis=2) == len(uuid)), numpy.logical_not(gs)).sum(axis=1) == 1)
    if show_wrong != '0':
      boring_inst = boring_inst_allright 
    else:
      boring_inst = numpy.logical_or(boring_inst_allright, boring_inst_allwrong)
    int_inst = numpy.logical_not(boring_inst)


    # Keep only interesting instances and interesting classes
    clabels = numpy.array(class_space)[int_cl]
    instlabels = numpy.array(instance_ids)[int_inst]
    classifs = classifs[:,int_cl,:]
    classifs = classifs[int_inst,:,:]
    gs = gs[int_inst,:]
    gs = gs[:,int_cl]

    # Compute confusion pairs
    cm_all = []
    for i in xrange(classifs.shape[2]):
      cl = classifs[:,:,i]
      cm = classification_matrix(gs, cl)
      cm_all.append(cm)
    cm_all = numpy.dstack(cm_all)

    pairs = {}
    for i, l_i in enumerate(clabels):
      for j, l_j in enumerate(clabels):
        if i != j:
          pairs[(i,j)] = list(cm_all[i,j,:])

    pairs_by_size = sorted(pairs, key=lambda x: sum(pairs[x]), reverse=True)

    info = {}
    info['class_space']            = md['class_space']
    info[ds_key]                   = md[ds_key]
    info['Total Classes']          = len(class_space)
    info['Interesting Classes']    = len(clabels)
    info['Total Instances']        = len(instance_ids)
    info['Interesting Instances']  = len(instlabels)

    # Compute the set of keys present in the metadata over all results 
    all_keys = sorted(reduce(set.union, (set(r.metadata.keys()) for r in results)))
    # Compute the set of possible values for each key 
    values_set = {}
    for k in all_keys:
      for r in results:
        try:
          values_set[k] = set(r.metadata.get(k,'UNKNOWN') for r in results)
        except TypeError:
          # skip unhashable
          pass
    # Compute the set of key-values which all the results have in common
    common_values = dict( (k, values_set[k].pop()) for k in values_set if len(values_set[k]) == 1)

    page = markup.page()
    page.init(**page_config)

    # Show summary
    page.h2('Summary')
    page.add(dict_as_html(info))

    # Statistical Tests
    # TODO: Add measures of correlation as well
    page.h2('Statistical Significance')
    if len(uuid) == 1:
      page.p("No test for single result")
    elif len(uuid) == 2:
      #page.p("McNemar's test")
      mcnemar_result = stats.mcnemar(self.interpreter, results[0], results[1])
      page.add(dict_as_html(dict(mcnemar=mcnemar_result)))
      mcnemar_pc = stats.mcnemar(self.interpreter, results[0], results[1], perclass=True)[int_cl]
      if goldstandard is not None:
        gs_i = list(clabels).index(goldstandard)
        page.add(dict_as_html({goldstandard:mcnemar_pc[gs_i]}))
      else:
        page.add(dict_as_html(dict(zip(clabels, mcnemar_pc))))
        
      # Added by Li Wang (li@liwang.info)
      randomisation_result = stats.randomisation(self.interpreter, results[0], results[1])
      page.add(dict_as_html(dict(randomisation=randomisation_result)))
    else:
      page.p("ANOVA")



    # Show common metadata
    page.h2('Common Metadata')
    page.add(dict_as_html(common_values))

    # Give the option to show/hide instances that are entirely wrong
    if show_wrong != '0':
      page.a('Hide Wrong', href='compare?' + urllib.urlencode({'uuid':uuid, 'show_wrong':0}, True))
    else:
      page.a('Show Wrong', href='compare?' + urllib.urlencode({'uuid':uuid, 'show_wrong':1}, True))

    # Confusion pairs tabulation
    with page.table:
      for key in values_set:
        # Display keys which differ
        if len(values_set[key]) > 1:
          with page.tr:
            page.th(key)
            page.td()
            for r in results:
              page.td(str(r.metadata.get(key,'-')))
      with page.tr:
        page.th()
        page.td()
        for id in uuid: page.th(id)

      # Display classification pairs
      for pair in pairs_by_size:
        if sum(pairs[pair]) == 0:
          # Can stop display, all the rest are zero
          break
        fr = clabels[pair[0]]  
        to = clabels[pair[1]] 
        if goldstandard is None or goldstandard == fr:
          with page.tr:
            label = ' => '.join((fr, to))
            page.th(label)
            page.td()
            for i, id in enumerate(uuid):
              with page.td:
                page.a(pairs[pair][i], href='classpair?'+urllib.urlencode({'gs':fr,'cl':to,'uuid':id}))

      # Display individual instances
      with page.tr:
        page.th()
        page.th('Goldstandard')
        for id in uuid: 
          with page.th:
            page.a(id, href='view?'+urllib.urlencode({'uuid':id}))

      for i, instance_id in enumerate(instlabels):
        inst_gs = gs[i]
        label = clabels[inst_gs]

        # Handle instance with no goldstandard labels
        if len(label) == 0: label = ''

        if goldstandard is None or goldstandard == label:
          with page.tr:
            with page.th:
              link = '../datasets/'+md[ds_key]+'/instances/'+instance_id
              page.a(instance_id, href= link)

            with page.td:
              page.a(label, href='compare?' + urllib.urlencode({'uuid':uuid, 'show_wrong':show_wrong, 'goldstandard':label}, True))
            for j, r_id in enumerate(uuid):
              inst_cl = classifs[i,:,j]
              labels = list(clabels[inst_cl])
              page.td(labels, **{'class':'correct' if (inst_gs==inst_cl).all() else 'wrong'})
      
    return str(page)

  @cherrypy.expose
  def delete(self, uuid, confirmed='N'):
    if self.store.mode != 'a':
      raise ValueError, "Store open in read-only mode"
    uuid = as_set(uuid)

    page = markup.page()
    page.init(**page_config)
    if confirmed == 'N':
      page.add("Delete the following results?")
      with page.ul:
        for id in uuid: page.li(uuid)
      page.a('YES', href='delete?' + urllib.urlencode({'uuid':uuid, 'confirmed':'Y'}, True))
    else:
      page.add("Deleted the folliwng results:")
      with page.ul:
        for id in uuid: 
          try:
            self.store._del_TaskSetResult(id)
            page.li('(Success) '+id)
          except KeyError:
            page.li('(Failure) '+id)
    return str(page)

  def get_full_summary(self, uuids, summary_fn = None):
    # Build the display summaries as we go, based on the stored summaries and any additional
    # summary function supplied.
    int_id = self.interpreter.__name__
    if summary_fn is None:
      summary_fn = self.summary_fn
    summaries = []

    for uuid in uuids:
      result = self.store._get_TaskSetResult(uuid)
      summary = result.summaries[int_id]
      missing_keys = set(summary_fn.keys) - set(summary)
      if len(missing_keys) > 0:
        result = self.store._get_TaskSetResult(uuid)
        summary_fn.init(result, self.interpreter)
        new_values = dict( (key, summary_fn[key]) for key in missing_keys )
        summary.update(new_values)
      summaries.append(summary)
    return summaries


  @cherrypy.expose
  def csv(self, uuid, columns=None):
    # TODO: Let user select columns
    # TODO: Apply the summary function. The summaries we get are straight from the store, and don't
    #       have the modifications resulting from browser-config
    uuid = as_set(uuid)
    int_id = self.interpreter.__name__
    fieldnames = zip(*self.relevant)[1]
    rows = self.get_full_summary(uuid)
    rows = project_compound(rows, fieldnames)
    
    import csv
    from cStringIO import StringIO
    out = StringIO()
    writer = csv.DictWriter(out, fieldnames, extrasaction='ignore')
    writer.writerows(rows)
    text = out.getvalue()
    cherrypy.response.headers["Content-Type"] = "text/csv"
    cherrypy.response.headers["Content-Length"] = len(text)
    cherrypy.response.headers["Content-Disposition"] = "attachment; filename=hydrat_browser.csv"
    return text
      

  @cherrypy.expose
  def metamap(self, uuid):
    import cgi
    import pprint
    uuids = as_set(uuid)
    int_id = self.interpreter.__name__
    map = metamap(self.store.get_Summary(uuid, int_id) for uuid in uuids)
    page = markup.page()
    page.init(**page_config)
    page.pre(cgi.escape(pprint.pformat(list(uuids))))
    page.add(dict_as_html(map))
    return str(page)
      
  @cherrypy.expose
  def view(self, uuid):
    """
    Display details for a single TSR
    """
    from hydrat.display.tsr import render_TaskSetResult
    
    tsr = self.store._get_TaskSetResult(uuid)
    confusion_matrix = tsr.overall_confusion_matrix(self.interpreter).sum(0)
    used_classes = (confusion_matrix[:,0] + confusion_matrix[:,3]) > 0
    summary = self.summary_fn(tsr, self.interpreter)
    from hydrat.result import CombinedMacroAverage, CombinedMicroAverage, PRF
    metric = PRF()

    rows = []
    footer = defaultdict(int)
    for i,cm in enumerate(confusion_matrix[used_classes]):
      row = {}
      #row['label'] = class_space[i]
      row['tp'], row['tn'], row['fp'], row['fn'] = cm
      row['precision'], row['recall'], row['fscore'] = metric(cm)
      footer['tp'] += row['tp']
      footer['tn'] += row['tn']
      footer['fp'] += row['fp']
      footer['fn'] += row['fn']
      rows.append(row)

    cols, col_headings = zip(*[
      ('tp',dict(label='TP', sorter='digit')),
      ('tn',dict(label='TN', sorter='digit')),
      ('fp',dict(label='FP', sorter='digit')),
      ('fn',dict(label='FN', sorter='digit')),
      ('precision', dict(label='P', sorter='digit')),
      ('recall', dict(label='R', sorter='digit')),
      ('fscore', dict(label='F', sorter='digit')),
      ])

    page = markup.page()
    page.init(**page_config)

    try:
      class_space = self.store.get_Space(tsr.metadata['class_space'])
    except NoData:
      class_space = ['class{0}'.format(i) for i in xrange(confusion_matrix.shape[0])]
      page.add('<h1>Warning: No data for class space "{0}"</h1>'.format(tsr.metadata['class_space']))

    # Data summmary
    page.add(dict_as_html(summary))

    # Overall summary
    space = list(numpy.array(class_space)[used_classes])
    page.dict_table(rows, cols, list(col_headings), space, footer)

    # Multiclass summary
    # TODO: If either the goldstandard or the output are multiclass,
    #       we will display a breakdown of the confusion matrix
    #       according to (x,y) -> (a,b,c).


    for i, result in enumerate(tsr.results):
      page.h2('Result %d' % i)
      cm_r = result.confusion_matrix(self.interpreter)
      used_classes = (cm_r[:,0] + cm_r[:,3]) > 0
      space_r = list(numpy.array(class_space)[used_classes])
      rows_r = []
      footer_r = defaultdict(int)
      for i, cm in enumerate(cm_r[used_classes]):
        row = {}
        row['tp'], row['tn'], row['fp'], row['fn'] = cm
        row['precision'], row['recall'], row['fscore'] = metric(cm)
        footer_r['tp'] += row['tp']
        footer_r['tn'] += row['tn']
        footer_r['fp'] += row['fp']
        footer_r['fn'] += row['fn']
        rows_r.append(row)
      footer_cm = footer_r['tp'],footer_r['tn'], footer_r['fp'], footer_r['fn']
      footer_r['precision'], footer_r['recall'], footer_r['fscore'] = metric(numpy.array(footer_cm))

      with page.table:
        with page.tr:
          with page.td:
            #TODO: More info here
            page.add(dict_as_html(result.metadata))
          with page.td:
            page.dict_table(rows_r, cols, list(col_headings), space_r, footer_r)
      



    #text = StringIO.StringIO()
    #with TableSort(text) as result_renderer:
    #  render_TaskSetResult(result_renderer, result, class_space, self.interpreter, summary)

    #page.add(text.getvalue())
    return str(page)

  @cherrypy.expose
  def confusion(self, uuid):
    """
    Display a sortable list of confusion pairs.
    """
    result = self.store._get_TaskSetResult(uuid)
    # TODO: make the class space a result attribute
    class_space = self.store.get_Space(result.metadata['class_space'])
    summary = self.summary_fn(result, self.interpreter)
    pairs = result.overall_classpairs(self.interpreter)

    rows = []
    fr_count = defaultdict(int)
    to_count = defaultdict(int)
    fp_count = defaultdict(int)
    fn_count = defaultdict(int)
    for cl_fr, cl_to in pairs:
      count = len(pairs[cl_fr, cl_to])
      fr_count[class_space[cl_fr]] += count
      to_count[class_space[cl_to]] += count
      if cl_fr != cl_to:
        fp_count[class_space[cl_to]] += count
        fn_count[class_space[cl_fr]] += count
        row = {}
        #TODO: Construct useful links here!!
        row['from'] = markup.oneliner.a(class_space[cl_fr], href='')
        row['to'] = markup.oneliner.a(class_space[cl_to], href='')
        link = 'classpair?'+urllib.urlencode({'uuid':result.uuid, 'gs':class_space[cl_fr], 'cl':class_space[cl_to]})
        row['count'] = markup.oneliner.a(count, href=link)
        rows.append(row)

    classes = sorted(map(class_space.__getitem__, set.union(*map(set,zip(*pairs)))))

    page = markup.page()
    page.init(**page_config)
    page.add(dict_as_html(summary))

    def highlight_zero(content):
      if content == 0:
        page.td(content, **{'class':'highlight'})
      else:
        page.td(content)

    with page.table:
      with page.tr:
        page.th('Class')
        for c in classes:
          link = 'classpair?'+urllib.urlencode({'uuid':result.uuid, 'gs':c, 'cl':c})
          page.th(markup.oneliner.a(c, href=link))
      with page.tr:
        page.th('Goldstandard')
        for c in classes:
          highlight_zero(fr_count[c])
      with page.tr:
        page.th('Classified')
        for c in classes:
          highlight_zero(to_count[c])
      with page.tr:
        page.th('False Positive')
        for c in classes:
          highlight_zero(fp_count[c])
      with page.tr:
        page.th('False Negative')
        for c in classes:
          highlight_zero(fn_count[c])
    page.dict_table(rows, ['from','to','count'], 
        col_headings=[
          {'label':'From', 'searchable':True}, 
          {'label':'To', 'searchable':True}, 
          'Count',
          ])
    return page()

  @cherrypy.expose
  def matrix(self, uuid, threshold=0):
    threshold = int(threshold)
    result = self.store._get_TaskSetResult(uuid)
    summary = self.summary_fn(result, self.interpreter)
    class_space = self.store.get_Space(result.metadata['class_space'])
    matrix = result.overall_classification_matrix(self.interpreter)
    matrix_sans_diag = numpy.logical_not(numpy.diag(numpy.ones(len(class_space), dtype=bool))) * matrix
    matrix_sans_diag *= matrix >= threshold
    interesting = numpy.logical_or(matrix_sans_diag.sum(axis=0), matrix_sans_diag.sum(axis=1))
    int_cs = numpy.array(class_space)[interesting]
    matrix = matrix[interesting].transpose()[interesting].transpose()

    page = markup.page()
    page.init(**page_config)
    page.add(dict_as_html(summary))
    with page.table:
      with page.tr:
        page.th()
        [ page.th(c) for c in int_cs ]
      for i, row in enumerate(matrix):
        with page.tr:
          page.th(int_cs[i])
          for j, val in enumerate(row):
            gs = int_cs[i]
            cl = int_cs[j]
            if val > threshold and gs != cl:
              td_attr={'class':'highlight'}
            else:
              td_attr={}
            page.td(**td_attr)
            link = 'classpair?'+urllib.urlencode({'uuid':uuid, 'gs':gs, 'cl':cl})
            page.a(str(val), href=link)
            page.td.close()

    return str(page)

  @cherrypy.expose
  def classpair(self, uuid, gs, cl):
    result = self.store._get_TaskSetResult(uuid)
    class_space = list(self.store.get_Space(result.metadata['class_space']))
    # TODO: handle dataset vs eval_dataset. Should everything have an eval_dataset?
    if 'eval_dataset' in result.metadata:
      dataset = result.metadata['eval_dataset']
    else:
      dataset = result.metadata['instance_space']
    docids = list(self.store.get_InstanceIds(dataset))
    pairs = result.overall_classpairs(self.interpreter)

    gs_i = class_space.index(gs)
    cl_i = class_space.index(cl)

    page = markup.page()
    page.init(**page_config)
    page.add(dict_as_html(result.metadata))
    page.h1("Classified from '%s' to '%s'" % (gs,cl))
    key = (gs_i, cl_i)
    tokenstreams = sorted(self.store.list_TokenStreams(dataset))
    featurespaces = sorted(self.store.list_FeatureSpaces(dataset))
    with page.table:
      for i in pairs[key]:
        with page.tr:
          id = docids[i]
          with page.th:
            page.a(id, href='../datasets/%s/instances/%s' % (dataset, id))
          for ts in tokenstreams:
            with page.td:
              page.a(ts,href='../datasets/%s/tokenstream/%s/%s' % (dataset, ts, id))
          for fs in featurespaces:
            with page.td:
              page.a(fs,href='../datasets/%s/features/%s/%s' % (dataset, fs, id))
    return str(page)

