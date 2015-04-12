import random
import csv
import time
import datetime
import os.path
import hydrat
from hydrat.common.pb import ProgressIter
from hydrat.common.mapmatrix import map2matrix 
from hydrat.store import Store
from hydrat.result.result import Result
from hydrat.result.tasksetresult import TaskSetResult
from hydrat.wrapper.langidnet import LangidNetLangid

def langidnet_langid(ds, tokenstream, key=None):
  cat = LangidNetLangid(apikey=key)
  ts = ds.tokenstream(tokenstream)
  filename = 'langidnet-%s-%s' % (tokenstream, ds.__name__)
  path = os.path.join(hydrat.config.getpath('paths','scratch'), filename)
  obtained = {}
  if os.path.exists(path):
    with open(path) as f:
      reader = csv.reader(f, delimiter='\t')
      for row in reader:
        obtained[row[0]] = row[1]

  interval = 3600.0 / cat.rate
  with open(path, 'a') as f:
    writer = csv.writer(f, delimiter='\t')
    for key in ProgressIter(ts, label='LangidNet-Langid'):
      if key in obtained: continue
      text = ts[key]
      now = datetime.datetime.now().isoformat()
      pred_lang = cat.classify(text)
      print pred_lang, now, text,
      writer.writerow((key, pred_lang, now, text.strip()))
      f.flush()
      time.sleep(interval) 
  return obtained

def do_langidnet(test_ds, tokenstream, class_space, classlabels, spacemap, key=None):
  raise DeprecationWarning("langid.net appears to be no longer operating")
  md = dict(\
    class_space  = class_space,
    dataset      = 'LangidNet',
    eval_dataset = test_ds.__name__,
    instance_space = 'LangidNet',
    eval_space   = test_ds.instance_space,
    learner      = 'LangidNet',
    learner_params = dict(tokenstream=tokenstream, spacemap=spacemap.__name__),
    )

  start = time.time()
  preds = langidnet_langid(test_ds, tokenstream, key=key)
  duration = time.time() - start
  for key in preds:
    preds[key] = [spacemap(preds[key])]

  cl = map2matrix( preds, test_ds.instance_ids, classlabels )
  gs = map2matrix( test_ds.classmap(class_space), test_ds.instance_ids, classlabels )

  result_md = dict(md)
  result_md['learn_time'] = 0.0
  result_md['classify_time'] = duration
  result = Result(gs, cl, test_ds.instance_ids, result_md )
  tsr = TaskSetResult( [result], md )
  return tsr

      
