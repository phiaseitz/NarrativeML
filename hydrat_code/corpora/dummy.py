from hydrat.dataset.text import ByteUBT
from hydrat.dataset.encoded import CodepointUBT, UTF8
from collections import defaultdict

class dummy(ByteUBT, CodepointUBT, UTF8):
  """Dummy backend for development use"""
  __name__ = "dummy"
  words = [u"test", u"exam", u"eggs", u"spam", u"blah"]

  def __init__(self, max_times = 100):
    self.__name__ += str(max_times)
    ByteUBT.__init__(self)
    CodepointUBT.__init__(self)
    self.max_times = max_times

  def ts_byte(self):
    docmap = {}
    for i in xrange(len(self.words)):
      for j in xrange(self.max_times):
        docmap["%04d"%(i * self.max_times + j)] = (self.words[i].encode('utf8') + " ") * (j+1)
    return docmap 

  def cm_dummy_default(self):
    classmap = {}
    for i in xrange(len(self.words)):
      for j in xrange(self.max_times):
        classmap["%04d"%(i * self.max_times + j)] = [u'class' + unicode(i) ]
    return classmap

  def cm_dummy_binary(self):
    classmap = {}
    for i in xrange(len(self.words)):
      for j in xrange(self.max_times):
        classmap["%04d"%(i * self.max_times + j)] = [u'class' + ('0' if i==0 else '1') ]
    return classmap

  def sp_dummy_default(self):
    return dict( train = self.instance_ids[:-(2*self.max_times)]
               , test = self.instance_ids[-(2*self.max_times):-self.max_times]
               , unused = self.instance_ids[-self.max_times:]
               )
    

class unicode_dummy(dummy):
  """Unicode dummy dataset"""
  __name__ = "dummy-unicode"
  words = [u"\N{POUND SIGN}pound",u'\N{BLACK STAR}blackstar',u'\N{WHITE STAR}whitestar',u'\N{LIGHTNING}lightning',u'\N{COMET}comet']

class single_char_dummy(dummy):
  __name__ = "dummy-single"
  words = [u'A',u'B',u'C',u'D',u'E']

class langid(unicode_dummy):
  """Dummy dataset with mapping into ISO639-1"""
  def cm_iso639_1(self):
    keys = self.cm_dummy_default().keys()
    return dict( (k, ['UNKNOWN']) for k in keys)

class unicode_dummy_multiclass(ByteUBT, CodepointUBT, UTF8):
  """Unicode dummy dataset with multiclass labels"""
  __name__ = "dummy-multiclass-unicode"
  words = [u"\N{POUND SIGN}pound",u'\N{BLACK STAR}blackstar',u'\N{WHITE STAR}whitestar']

  def __init__(self, max_times = 100):
    ByteUBT.__init__(self)
    CodepointUBT.__init__(self)
    self.max_times = max_times
    self.__name__ += str(max_times)

  def ts_byte(self):
    docmap = {}
    for i in xrange(len(self.words)):
      for j in xrange(self.max_times):
        docmap["%04d"%(j)+str(i)] = (self.words[i].encode('utf8') + " ") * (j+1)
    for i in xrange(len(self.words)):
      for j in xrange(len(self.words)):
        for n in xrange(self.max_times):
          docmap["%04d"%(n)+str(i)+str(j)] = (self.words[i].encode('utf8')+" "+self.words[j].encode('utf8')+" ")*(n+1)
    for n in xrange(self.max_times):
      docmap["%04d"%(n)+"012"] = (" ".join(self.words[i].encode('utf8') for i in xrange(len(self.words)))+" ")*(n+1)
    return docmap 

  def cm_dummy_multiclass(self):
    classmap = {}
    for i in xrange(len(self.words)):
      for j in xrange(self.max_times):
        classmap["%04d"%(j)+str(i)] = [u'class' + unicode(i)]
    for i in xrange(len(self.words)):
      for j in xrange(len(self.words)):
        for n in xrange(self.max_times):
          classmap["%04d"%(n)+str(i)+str(j)] = [u'class'+unicode(i), u'class'+unicode(j)]
    for n in xrange(self.max_times):
      classmap["%04d"%(n)+"012"] = [u'class'+unicode(i) for i in xrange(len(self.words))]
    return classmap
  
  
