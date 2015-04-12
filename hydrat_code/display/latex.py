def latex_escape(input):
  escaped_symbols = '_'
  output = ''
  for c in input:
    if c in escaped_symbols:
      output += '\\' + c
    else:
      output += c
  return output

def latex_format(entries):
  output = []
  for entry in entries:
    if isinstance(entry, int):
      output.append(str(entry))
    elif isinstance(entry, float):
      output.append('%.3f' % entry)
    elif isinstance(entry, str):
      output.append(latex_escape(entry))
    else:
      output.append(str(entry))
  return output

def latex_table_row(entires, cell_renderer = latex_format):
  return ' & '.join(cell_renderer(entires)) + r'\\' + '\n'

def latex_table(matrix, header, cell_renderer = latex_format):
  output = r'\begin{tabular}' + '{|%s|}' % '|'.join('c' for x in header)
  output += r'\hline'+'\n'
  output += latex_table_row(header)
  output += r'\hline'+'\n'
  for row in matrix:
    output += latex_table_row(row, cell_renderer = latex_format)
  output += r'\hline'+'\n'
  output += r'\end{tabular}'+'\n'
  return output

def dict_table(rows, col_ids, cell_renderer = latex_format):
  matrix = []
  headers, entries = zip(*col_ids)
  for row in rows:
    matrix.append([ row[e] for e in entries ])
  return latex_table(matrix, headers)
