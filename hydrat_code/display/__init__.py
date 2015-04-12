
def show_metadata(renderer, metadata):
  entries = []
  for key in sorted(metadata):
    entries.append({'key':key , 'value':str(metadata[key])})
  renderer.dict_table( entries
                     , ["key", "value"]
                     , col_headings = ["Key", "Value"]
                     , title = "Metadata"
                     )
  
