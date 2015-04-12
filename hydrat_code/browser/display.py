import hydrat.common.markup as markup

def list_of_links(assoc_list):
  page = markup.page()
  page.ul()
  for k,v in assoc_list:
    page.li()
    page.a(k, href=v) 
    page.li.close()
  page.ul.close()
  return str(page)

def list_as_html(l):
  page = markup.page()
  page.ul()
  for text in l:
    page.li()
    page.add(text)
    page.li.close()
  page.ul.close()
  return str(page)

def dict_as_html(d):
  page = markup.page()
  page.table()
  for key in sorted(d):
    page.tr()
    page.td(key)
    page.td(str(d[key]))
    page.tr.close()
  page.table.close()
  return str(page)

def dict_table( rows
              , col_keys
              , col_headings  = None
              , row_headings  = None
              , title         = None
              , default       = None
              ):
  if col_headings and len(col_headings) != len(col_keys):
    raise ValueError, "Column headings do not match column keys"
  if row_headings and len(row_headings) != len(rows):
    raise ValueError, "Row headings do not match rows"
  page = markup.page()
  with page.table:
    page.caption(title)

    if row_headings and col_headings:
      col_headings = [''] + col_headings
    with page.tr:
      for h in col_headings: page.th(h)

    for i, row in enumerate(rows):
      with page.tr:
        if row_headings: page.th(row_headings[i])
        for key in col_keys: 
          datum = row[key] if key in row else default
          page.td(datum)

  return str(page)
