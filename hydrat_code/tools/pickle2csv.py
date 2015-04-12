"""
Simple example of how to process the summaries.pickle file produced along with the summary pages.
This script just converts the pickle into csv format. You are encouraged to process the data in the
pickle file directly, rather than converting to csv and parsing it back.
"""
from cPickle import load
from csv import DictWriter

# Columns to produce. The uncommented ones are the most commonly used ones. 
cols =\
 [ 'dataset'
 #, 'date'
 #, 'uuid'
 #, 'name'
 #, 'dataset_uuid'
 #, 'feature_uuid'
 #, 'class_uuid'
 #, 'seed'
 #, 'interpreter'
 #, 'probabilisitc'
 #, 'task_type'
 , 'class_name'
 , 'feature_desc'
 , 'learner'
 , 'learner_params'
 , 'micro_fscore'
 , 'micro_precision'
 , 'micro_recall'
 , 'macro_fscore'
 , 'macro_precision'
 , 'macro_recall'
 , 'avg_learn'
 , 'avg_classify'
 #, 'link'
 #, 'perfold_macroprf'
 #, 'perfold_microprf'
 ]

def csv_output(file, summaries):
  """
  Write the summaries to an open file object.
  """
  writer = DictWriter(file, cols, extrasaction='ignore', restval='UNKNOWN')
  writer.writerows(summaries)

if __name__ == "__main__":
  with open('summaries.pickle') as f:
    summaries = load(f)
  #print reduce(set.union, map(set ,map(dict.keys, summaries)))
  with open('summaries.csv','w') as outf:
    csv_output(outf, summaries)
