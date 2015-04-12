from cStringIO import StringIO
from hydrat.display.tsr import result_summary_table
from hydrat.display.html import TableSort

def results2html(store, browser_config, tsr_metadata=None):
  if tsr_metadata is None:
    tsr_metadata = {}

  interpreter = browser_config.interpreter
  summary_fn  = browser_config.summary_fn
  int_id = interpreter.__name__

  summaries = []
  for tsr in store.get_TaskSetResults(tsr_metadata):
    summary = tsr.summarize(summary_fn, interpreter)
    summaries.append(summary)

  io = StringIO()
  with TableSort(io) as renderer:
    result_summary_table(summaries, renderer, relevant = browser_config.relevant)
  return io.getvalue()
