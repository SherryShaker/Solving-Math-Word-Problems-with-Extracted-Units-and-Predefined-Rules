## Expression Generation for Solving MWPs Constrained by Extracted Units and Predefined Rules

### **Project Introduction**
---
Systems designed to solving Math Word Problems (MWPs) automatically accept math problems described with natural language, and convert or map the inputs to an equivalent structured expression to deliver final answers. Recently, researchers shift their direction and focus on applying deep learning architectures, such as seq2seq and transformer models, on MWPs solving without manual feature engineering. Despite deep neural networks (DNNs) could provide fair results on this task considering their strong generalisation ability, they are proved by evidence to rely on shallow heuristics to perform well without truly “understand” the problems. Motivated by this issue, we proposed to include unit constraints and inference while training DNNs to generate equation. With the help of extracted units and context information relative to quantities, the model might gain a better “understanding” to both the question and premise part of a MWP. Nevertheless, before the development of our designed system, we have to build our own information extraction system in light of the unsatisfactory performance of the off-the-shelf tools on the MWP datasets and few research regarding the extraction of context information, such as measured entities, properties, and qualifiers. 

Therefore, this project could be divided into two phase: 

1) The design for deep-learning-based equation generation restricted by extracted units and predefined rules. 

2) The design for quantity and unit extraction algorithm along with implementation.

### **Requirements**
---
flair==0.11.3
torch==1.12.0

### **File Structure**
---
```
.
│  README.md
│  
├─NER_task         /*predict annotated span types*/
│  │  ner_entity.sh
│  │  NER_task.ipynb
│  │  ner_unit.job
│  │  predicting_ner.py     /*basic span types prediction*/
│  │  predicting_ner_unit.py    /*unit span prediction*/
│  │  test_data.ipynb           /*data construction*/
│  │  train_data.ipynb
│  │  trial_data.ipynb
│  │  
│  ├─.ipynb_checkpoints
│  │      NER_task-checkpoint.ipynb
│  │      test_data-checkpoint.ipynb
│  │      train_data-checkpoint.ipynb
│  │      trial_data-checkpoint.ipynb
│  │      
│  ├─data_entity    /*annotated data for basic span type*/
│  │      dev.txt
│  │      test.txt
│  │      train.txt
│  │      
│  └─data_unit      /*annotated data for unit type*/
│          dev.txt
│          test.txt
│          train.txt
│          
└─question_binary classifier/*classify question and premise*/
    │  binary_classifier_v2.ipynb   /*train a classifier*/
    │  dataset.ipynb                /*construct dataset*/
    │  embeds.pkl
    │  interrogative_binary classifier.ipynb
    │  
    ├─.ipynb_checkpoints
    │      binary_classifier_v2-checkpoint.ipynb
    │      dataset-checkpoint.ipynb
    │      interrogative_binary classifier-checkpoint.ipynb
    │      
    ├─AQuA      /*AQuA MWP dataset*/
    │      dev.json
    │      test.json
    │      train.json
    │      
    ├─classify_ques_data    /*combined dataset*/
    │      test.json
    │      train.json
    │      
    └─online_classify_data  /*Kaggle dataset*/
            test.csv
            train.csv
            val.csv