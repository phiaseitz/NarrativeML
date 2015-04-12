from __future__ import with_statement

import os
import numpy
numpy.seterr(all='raise')

from hydrat.result import CombinedMacroAverage, CombinedMicroAverage, PRF
from hydrat.display import show_metadata
from hydrat.display.html import table_str, HTMLWriter, TableSort

from logging import getLogger

logger = getLogger('hydrat.display.tsr')
KEY_SEP =':'

def project_compound(summaries, cols):
  # Process compound keys, which are meant to project from a dict metadata value
  for col in cols:
    if KEY_SEP in col:
      keys = col.split(KEY_SEP)
      for s in summaries:
        v = s
        try:
          for k in keys:
            try:
              v = v[k]
            except TypeError:
              v = v[int(k)]
        except (KeyError,TypeError):
          v = None
        s[col] = v
  return summaries

def summarize_TaskSetResult(result, interpreter):
  raise NotImplementedError, "Stop using this! Use hydrat.display.summary_fn"

def result_summary_table(summaries, renderer, relevant = None, title = None):
  if relevant is None:
    relevant = [(k.title(),k) for k in sorted(summaries[0].keys()) if not k.startswith('_')]

  headings, cols = zip(*relevant)
  summaries = project_compound(summaries, cols)
      
  renderer.dict_table( summaries 
                     , cols 
                     , col_headings = headings 
                     , title = title 
                     )


def render_TaskSetResult(renderer, tsr, classlabels, interpreter, summary):
  show_metadata(renderer, summary)
  render_TaskSetResult_summary(renderer, tsr, classlabels, interpreter)
  for i,result in enumerate(tsr.results):
    renderer.subsection("Result %d"% i)
    show_Result(renderer, result, interpreter)

def render_TaskSetResult_summary(renderer, tsr, classlabels, interpreter):
  confusion_matrix = tsr.overall_confusion_matrix(interpreter)
  classification_matrix = tsr.overall_classification_matrix(interpreter)
  display_confusion_matrix = confusion_matrix.sum(axis=0)

  row_labels = numpy.array(classlabels)

  # confusion matrix columns
  # Calculate a totals row for the confusion matrix
  totals = display_confusion_matrix.sum(axis=0)
  cm_tab = ( display_confusion_matrix
           , [ 'tp', 'tn', 'fp', 'fn' ]
           , map(renderer.text_method, totals)
           )

  # class counts
  count_matrix = numpy.array([[row[0]+row[3]] for row in display_confusion_matrix])
  count_tab = ( count_matrix
              , [ 'N' ]
              , [ renderer.text_method(count_matrix.sum()) ]
              )

  # PRF columns
  metric = PRF()
  prf_matrix = numpy.array([metric(row) for row in display_confusion_matrix])

  prf_tab = ( prf_matrix 
            , [ 'P', 'R', 'F' ]
            , map( renderer.text_method, metric(totals) )
            )

  # classification matrix
  classif_tab = ( classification_matrix
                , classlabels 
                , map(renderer.text_method, classification_matrix.sum(axis=0))
                )

  matrix, head, foot = zip(count_tab, cm_tab, prf_tab, classif_tab)
  head = sum(head, [])
  foot = sum(foot, [])

  # Hack to work around numpy's requirement of a uniform type for arrays.
  # We render everything into a numpy string array first.
  string_matrix = []
  for m in matrix:
    s = numpy.zeros(m.shape, dtype='S64')
    for r in range(m.shape[0]):
      for c in range(m.shape[1]):
        s[r,c] = renderer.text_method(m[r,c])
    string_matrix.append(s)
    
  matrix = numpy.hstack(string_matrix)

  renderer.array_table( matrix
                      , col_headings = head
                      , row_headings = row_labels
                      , col_footings = foot
                      )

  
def show_Result(renderer, result, interpreter):
  renderer.target.write('<table cellspacing=15><tr>')
  renderer.target.write('<td>')
  show_metadata(renderer, result.metadata)
  renderer.target.write('</td>')
  renderer.target.write('<td>')
  show_ClassificationMatrix(renderer, result, interpreter)
  renderer.target.write('</td>')
  renderer.target.write('<td>')
  show_ConfusionMatrix(renderer, result, interpreter)
  renderer.target.write('</td>')
  renderer.target.write('</tr></table>')

def show_ConfusionMatrix(renderer, result, interpreter):
  row_headings = None
  if hasattr(result, 'classlabels'):
    row_headings = list(result.classlabels)
  renderer.array_table( result.confusion_matrix(interpreter) 
                      , col_headings = ['TP','TN','FP','FN']
                      , row_headings = row_headings
                      , title = "Confusion Matrix"
                      )

def show_ClassificationMatrix(renderer, result, interpreter):
  def custom_cell(text, type):
    if text == 0.0:
      text = ''
    if isinstance(text, float):
      text = int(text)
    else:
      text = str(text)
    renderer._openTag(type)
    renderer._text(text)
    renderer._closeTag(type)

  row_headings = None
  col_headings = None
  if hasattr(result, 'classlabels'):
    row_headings = list(result.classlabels)
    col_headings = list(result.classlabels)

  original_cell = renderer._cell
  renderer._cell = custom_cell
  renderer.array_table( result.classification_matrix(interpreter) 
                      , col_headings = col_headings
                      , row_headings = row_headings
                      , title = "Classification Matrix"
                      )
  renderer._cell = original_cell 
