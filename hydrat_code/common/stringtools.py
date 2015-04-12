import re
##
# Removes HTML markup from a text string.
# from http://effbot.org/zone/re-sub.htm#strip-html
#
# @param text The HTML source.
# @return The plain text.  If the HTML source contains non-ASCII
#     entities or character references, this is a Unicode string.

def strip_html(text):
    def fixup(m):
        text = m.group(0)
        if text[:1] == "<":
            return "" # ignore tags
        if text[:2] == "&#":
            try:
                if text[:3] == "&#x":
                    return unichr(int(text[3:-1], 16))
                else:
                    return unichr(int(text[2:-1]))
            except ValueError:
                pass
        elif text[:1] == "&":
            import htmlentitydefs
            entity = htmlentitydefs.entitydefs.get(text[1:-1])
            if entity:
                if entity[:2] == "&#":
                    try:
                        return unichr(int(entity[2:-1]))
                    except ValueError:
                        pass
                else:
                    return unicode(entity, "iso-8859-1")
        return text # leave as is
    return re.sub("(?s)<[^>]*>|&#?\w+;", fixup, text)#

import string
# Removes punctuation. may be wise to do this -after- stripping HTML.
def strip_punctuation(text):
  output = ''
  for char in text:
    if char not in string.punctuation:
      output += char
  return output 
