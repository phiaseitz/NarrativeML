from hydrat.summary import classification_summary
summary_fn = classification_summary()

from hydrat.result.interpreter import SingleHighestValue 
interpreter = SingleHighestValue()

relevant  = [
  ( {'label':"Dataset", 'searchable':True}       , "metadata:dataset"       ),
  ( {'label':"Class Space",'searchable':True}     , "metadata:class_space"     ),
  ( {'label':"Feature Desc",'searchable':True}   , "metadata:feature_desc"     ),
  ( {'label':"Learner",'searchable':True}    , "metadata:learner"    ),
  ( {'label':"Params",'searchable':True}    , "metadata:learner_params"    ),
  ( "Macro-F"       , "macro_fscore"        ),
  ( "Macro-P"     , "macro_precision"     ),
  ( "Macro-R"        , "macro_recall"        ),
  ( "Micro-F"       , "micro_fscore"        ),
  ( "Micro-P"     , "micro_precision"     ),
  ( "Micro-R"        , "micro_recall"        ),
  ( {'sorter':'digit', 'label':"Learn Time"}    , "metadata:avg_learn"     ),
  ( {'sorter':'digit', 'label':"Classify Time"} , "metadata:avg_classify"  ),
]
