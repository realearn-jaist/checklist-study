# Introduction:
This repository is based on the checklist-study presented in the paper https://aclanthology.org/2020.acl-main.442/. It provides implementations and test runs for various NLP tasks, including sentiment analysis, question answering, and Quora question pair detection. Each folder contains resources for creating and testing these NLP tasks. The repository tests multiple capabilities, with each capability assessed using specific test types: Minimum Functionality Tests (MFT), Diagnostic Tests (DIR), and Invariance Tests.


## Installation

To ensure compatibility, it's recommended to use Python version 11 in the path because the munch library supports a maximum of Python version 11, and munch library does not support version 3.12.

If you encounter issues compiling cells after installation, please restart the kernel or the Anaconda command prompt.

Clone the author's repository

```
git clone https://github.com/marcotcr/checklist
cd checklist
pip install -e .
```

Install torch (required)

```pip install torch```

Install the required packages

```pip install spacy```

Download the English language model for spaCy

```python -m spacy download en_core_web_sm```

Install TensorFlow 

```pip install tensorflow```

Install munch

```pip install munch```

Install pattern 
used in qqp

```pip install pattern```

Download the English module for pattern

```python -m pattern.en.download```


## Overview

Every NLP task, including sentiment analysis, Quora question pair matching, and the Stanford Question Answering Dataset, is evaluated using the following tests: Minimum Functionality Test (MFT), Invariance Test (INV), and Directional Expectation Test (DIR).

- **MFT (Minimum Functionality Test):** Focuses on evaluating whether a model has the basic. Unit test in software engineering
functionality.
- **DIR (Directional Expectation Test):** Determines whether a model’s predictions are consistent with a prior expectation or hypothesis. Expect the label to change in a certain way.
- **INV (Invariance Test):** Checks whether a model is invariant to certain transformations or changes in the input data. Input the perturbation and expect model prediction to remain the same

To assess various capabilities for each NLP task, the following aspects are tested:

- **Vocabulary + Parts of Speech (POS):** Important words/word types for the task.
- **Taxonomy:** Synonyms/antonyms, superclasses, and subclasses.
- **Robustness:** Resilience to typos and irrelevant changes.
- **Named Entity Recognition (NER):** Appropriately understanding named entities given the task.
- **Temporal:** Understanding the order of events.
- **Negation:** Ability to detect changes in sentiment, e.g., through negation.
- **Coreference:** Understanding which entities are referred to by pronouns like “his / her” or terms like “former / latter.”
- **Semantic Role Labeling (SRL):** Understanding roles (role and subject).
- **Logic:** Ability to handle symmetry, consistency, and conjunctions.
- **Fairness:** Ensuring equitable performance across different contexts, which can vary in meaning depending on the application. [2]

### qqp-bert-roberta
The purpose of this folder is to facilitate the creation of Quora question pair test cases and to test the test suite. The `test-qqp-bert` suite uses the `bert-base-cased-finetuned-mrpc` model from Hugging Face, while the `test-qqp-roberta` suite uses the `JeremiahZ/roberta-base-mrpc` model from Hugging Face. These tests utilize the Quora question pair dataset to determine whether a given pair of questions is a duplicate or not, as assigned by the author.

    
- **create-QQP.ipynb**
    
    This notebook is used to prepare the Quora Question Pairs (QQP) test.  After processing the test suite, it will save to the `qqp_suite.pkl` file which will be used to run the test case in `test-qqp-bert` and `test-qqp-roberta`. The Quora question pair notebook usually tests the 2 questions to check that both of them are equal or not, author provides different tests, such as a test on vocab of 2 questions, which have the same generate the text question pair with the place holder `first_name` `last_name` `adj` `noun` and the result label expected should be non-duplicate if its results label as 1 it will be the fail test case. The capabilities that were tested on this NLP task are capability, taxonomy, ner (name entity recognition), temporal, negation, core, SRL (Semantic Role Labeling), and logic. Except the fairness.

- **dev.tsv**

    The quora question pair dataset.

- **qqp_suite.pkl**

    An object file that contains the test suite for evaluating the QQP model including test cases and scenarios to assess model performance.

- **test-qqp-bert.ipynb**
    This notebook is used to test the performance of the BERT model which model name `bert-base-cased-finetuned-mrpc`on the QQP dataset. It includes loading the pre-trained BERT model, running it on the test data, and evaluating the results. 

- **test-qqp-roberta.ipynb**
    As mentioned earlier, this one is used to test the performance of the RoBERTa model which is `JeremiahZ/roberta-base-mrpc` model on the QQP dataset. It involves loading the pre-trained RoBERTa model, running it on the test data, and analyzing the outcomes. 

### sentimental-revise 

This folder provided the revision of creating and testing on sentimental analysis of the author's repository. with the airline tweets dataset. The test scenario mainly about the feedback on airline sentiment. 

- **create-sentimental-test.ipynb**

    This notebook is responsible for creating test cases for sentiment analysis. After processing all test suites, it will save the test cases to sentimental_suite.pkl. For example, when an intensifier is added, the confidence should not decrease. Similarly, when focusing on a specific task, if the phrase "used to" is included, the confidence should not increase.

    This sentiment analysis NLP task tests the following capabilities:

    - Capability: Vocabulary
        - MFTs focus on basic function
        - Intensifiers and reducers
        - INVariance: change neutral words
        - Add negative and positive phrases
    - Capability: Robustness
        - INVariance: adding irrelevant content before and after
        - Punctuation, contractions, typos
    - Capability: NER
    - Capability: Temporal Awareness
    - Capability: Fairness
    - Capability: Negation
    - Capability: SRL (Semantic Role Labeling)

- **sentimental_suite.pkl**
    An object file containing the test suite for evaluating the sentiment analysis model including test cases and evaluation metrics.
- **test-sentimental.ipynb**
    This notebook is used to test the sentiment analysis model. It involves loading the pretrained model (distilbert/distilbert-base-uncased-finetuned-sst-2-english from huggingface), running it on the test dataset, and evaluating its performance.
- **Tweets.csv**
    This is CSV file containing tweet data used for sentiment analysis.

### sentimental-scenario-1

This folder provided the revision of creating and testing on sentimental analysis on the amazon sentiment dataset custormer's review product.

- **create-sentimental-dt1.ipynb**
    This notebook is responsible for creating test cases for sentiment analysis. After processing all test suites, it will save the test cases to sentimental_suite_dt1.pkl. This scenario had changed some word from the author to approprate with the amazon dataset for example, `This was a well aircraft` to `This was a well silent movie.`. This scenario nlp task also used on the following capabilities: 
    - Capability: Vocabulary
    - Capability: Temporal Awareness
    - Capability: Negation
    - Capability: SRL

- **sentimental_suite_dt1.pkl**
    An object file containing the test suite for evaluating the sentiment analysis model including test cases and evaluation metrics.
    
- **test-sentimental-dt1.ipynb**
    This notebook is used to test the sentiment analysis pipeline.

- **amazon_review.csv**
    This is CSV file containing product's review on amazon service used for sentiment analysis.

### sentimental-scenario-2
This folder provides the revision for creating and testing sentiment analysis on climate sentiment. For example, in sentiment scenario 2:
- 'Risk': 0
- 'Opportunity': 2
- 'Neutral': 1
These categories are used to evaluate climate change sentiment as well as simple sentiment analysis. "Risk" refers to the potential for negative events that can pollute the climate. "Opportunity" refers to the potential for actions that can decrease climate pollution. "Neutral" means there is no significant sentiment expressed.
    
- **create-sentimental-dt2.ipynb**

    This notebook is responsible for creating test cases for sentiment analysis. After processing all test suites, it will save the test cases to sentimental_suite_dt2.pkl, and this notebook is used to create and preprocess a second specific sentiment analysis dataset espescialy from climate-related tweets. Test on following capabilities
    - Capability: Vocabulary
        - MFTs
        - Intensifiers and reducers
        - Add positive phrases
    - Capability: Robustness
        - punctuation, contractions, typos
    - Capability: Temporal Awareness
    - Capability: Negation
    - Capability: SRL

- **sentimental_suite_dt2.pkl**
    
    An object file containing the test suite for evaluating the sentiment analysis model including test cases and evaluation metrics.

- **test-sentimental-dt2.ipynb**
    This notebook is used to test the sentiment analysis pipeline.

- **climate.csv**
     This is CSV file containing climate sentimental analysis.

### squad-revise
This folder is a revision of the author's repository. This NLP task mainly used the editor.mask which is in the checklist library to generate the sentence to test on each capability and keep on the test case. However, this task also use some of dataset about SQuAD from the Datasets Library (python library).

- **SQuAD-create-test-suite.ipynb**
    This notebook is used to create the test suite for the SQuAD (Stanford Question Answering Dataset) task. It involves steps like selecting and preprocessing the data and defining test cases. For example, this test case will provide the context which meant some information, and author needs to provide question and exactly answer. and use model question-answering that has defined above,  `predconfs` function to provide and answer. and in this file has provided short summary of each test case. 
    
    As an example of this, 
    ```
    C: Laura became a waitress before Roy did.
    Q: Who became a waitress last?
    A: Roy
    P: Laura
    ```

    Above is an example of fail test case, you can see that the context is "Laura became a waitress `before` Roy did.". The question is "Who became a waitress `last`?". The context and question are different is before and last, so the expected answer is "Roy", but the predicted answer from the model is "Laura". So mean that the question-answering model isn't understand the word before and after.
    This task has tested on many capabilities.
    
    - Capability: Vocabulary
    - Taxonomy: 
        - Size, shape, color, age, material
        - Professions vs nationalities
        - Animal vs vehicle
    - Robustness
    - NER
    - Temporal
    - Negation
    - Fairness spinoff
    - Coreference (Understanding which entities are referred to by “his / her”, “former / latter”)
    - Semantic Role Labeling (SRL) (Understanding roles (Role and subject))

    This notebook is responsible for creating test cases for SQuAD. After processing all test suites, it will save the test cases to squad_suite.pkl

- **SQuAD-create-run-suite.ipynb**
    This notebook is used to run the SQuAD test suite. It includes loading the question answering pipeline, running the test cases, and summarizing the model's test suite.

- **squad_suite.pkl**
    An object file containing the test suite for evaluating the question answering model including test cases and evaluation metrics.

## Appendix

### Dataset Source
- revise sentiment: [Twitter Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- sentiment 1: [Mini Amazon Sentimental Dataset](https://huggingface.co/datasets/GerindT/mini_amazon_sentimental/tree/main)
- sentiment 2: [Climate Sentiment Dataset](https://huggingface.co/datasets/climatebert/climate_sentiment/blob/main/data/train-00000-of-00001-04b49ae22f595095.parquet)

### Model Source:
- qqp: [BERT Base Cased Fine-tuned on MRPC](https://huggingface.co/bert-base-cased-finetuned-mrpc) model
- qqp roberta: [RoBERTa Base Fine-tuned on MRPC](https://huggingface.co/JeremiahZ/roberta-base-mrpc) model


### Any additional information goes here

[1] https://www.godeltech.com/how-to-automate-the-testing-process-for-machine-learning-systems/ 

[2] https://blog.allenai.org/using-checklists-with-allennlp-behavioral-testing-of-textual-entailment-models-1a0aa43cdb28
