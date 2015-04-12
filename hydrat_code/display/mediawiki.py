def tostr(obj):
  if type(obj) == float:
    return "%1.5f"%obj
  else:
    return str(obj)
    
class mediawiki:
  def __init__(self, target, depth = 0):
    self.target = target
    self.depth = depth

  def _get_depth(self): return self.depth
  def _set_depth(self, new_depth): self.depth = new_depth
  depth = property(_get_depth, _set_depth)

  def write(self, str): self.target.write(str+"\n")

  def writeln(self, str): self.target.write(str+"\n\n")

  def section(self, str, depthplus = 0):
    equals_part = "=" * (2 + self.depth + depthplus)
    self.write("%s %s %s" % (equals_part, str, equals_part))

  def column_headings(self, labels):
    self.write("! " + " !! ".join(labels))

  def row_heading(self, label):
    self.write("! " + label)

  def row(self, entries, label = None):
    self.write("|-")
    if label: self.row_heading(label)
    self.write("| " + " || ".join([tostr(entry) for entry in entries]))

  def title(self, msg):
    self.write("|+ %s" % msg)

  def display_classification_matrix(self, matrix, labels):
    self.write("{|")
    self.title("gold on left, assigned on top")
    self.column_headings(["&times;"] + labels)
    for from_label in labels:
      self.row([ matrix.get((from_label,to_label), 0) for to_label in labels], label = from_label)
    self.write("|}")
    
  def display_cooccurance_matrix(self, matrix, labels):
    self.write("{|")
    self.title("base on left, co-occurance on top")
    self.column_headings(["&times;"] + labels)
    for from_label in labels:
      self.row( [ 
                 ( "&times;", matrix.get((from_label,to_label), 0) )[from_label!=to_label]
                  for to_label in labels
                ]
                , label = from_label
              )
    self.write("|}")
        
        
  def display_table(self, rows, title=None, headings=None, row_headings = None):
    if row_headings: assert len(rows) == len(row_headings)
    self.write('{| border = "1"')
    if title: self.title(title)
    if headings and row_headings: self.column_headings([""]+headings)
    elif headings: self.column_headings(headings)
    if not row_headings: row_headings = [None] * len(rows)
    for (entries, label) in zip(rows, row_headings):
      self.row(entries, label)
    self.write("|}")

  def display_dict_table(self, rows, cols, headings=None, title=None):
    if headings: assert len(headings) == len(cols)
    self.write('{| border = "1"')
    if title: self.title(title)
    if headings: self.column_headings(headings)
    for entry in rows:
      _row = []
      for c in cols:
        if c in entry:
          _row.append(entry[c])
        else:
          _row.append("UNKNOWN")
      self.row(_row)
    self.write("|}")
