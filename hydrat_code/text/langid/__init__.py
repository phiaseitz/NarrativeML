#from textcat import TextCat
from langdetect import LangDetect
#from google import GoogleLangid
#from bing import BingLangid
try:
  from langiddotpy import LangidDotPy
except ImportError:
  pass

try:
  from chromecld import ChromeCLD
except ImportError:
  pass

try:
  from libtextcat import LibTextCat
except ImportError:
  pass
from lastrings import LAStrings
