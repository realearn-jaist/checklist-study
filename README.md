**Introduction:**

needs to install the following library.

To ensure compatibility, it's recommended to use Python version 11 in the path because the munch library supports a maximum of Python version 11 because munch library does not support version 3.12.

If you encounter issues compiling cells after installation, please restart the kernel or the Anaconda command prompt.
# install checklist
```
git clone https://github.com/marcotcr/checklist
cd checklist
pip install -e .
```

suite.visual_summary_table()
# Install torch (required)
```pip install torch```

# Install the required packages
```pip install spacy```

# Download the English language model for spaCy
```python -m spacy download en_core_web_sm```



# Update Python
# Install TensorFlow 
```pip install tensorflow```

# Install munch
```pip install munch```

# Install pattern 
used in qqp
```pip install pattern```

# Download the English module for pattern

```python -m pattern.en.download```


# Index
- qqp-bert-roberta
    - create-QQP.ipynb
    - dev.tsv
    - qqp_suite.pkl
    - test-qqp-bert.ipynb
    - test-qqp-roberta.ipynb
- sentimental-revise 
    - create-sentimental-test.ipynb
    - sentimental_suite.pkl
    - test-sentimental.ipynb
    - Tweets.csv
- sentimental-scenario-1
    - create-sentimental-dt1.ipynb
    - sentimental_suite_dt1.pkl
    - test-sentimental-dt1.ipynb
    - amazon_review.csv
- sentimental-scenario-2
    - create-sentimental-dt2.ipynb
    - sentimental_suite_dt2.pkl
    - test-sentimental-dt2.ipynb
    - climate.csv
- squad-revise
    - SQuAD-create-test-suite.ipynb
    - SQuAD-create-run-suite.ipynb
    - squad_suite.pkl


# Dataset Source
- revise sentiment: [Twitter Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- sentiment 1: [Mini Amazon Sentimental Dataset](https://huggingface.co/datasets/GerindT/mini_amazon_sentimental/tree/main)
- sentiment 2: [Climate Sentiment Dataset](https://huggingface.co/datasets/climatebert/climate_sentiment/blob/main/data/train-00000-of-00001-04b49ae22f595095.parquet)

# Model Source:
- qqp: [BERT Base Cased Fine-tuned on MRPC](https://huggingface.co/bert-base-cased-finetuned-mrpc) model
- qqp roberta: [RoBERTa Base Fine-tuned on MRPC](https://huggingface.co/JeremiahZ/roberta-base-mrpc) model


Chommakorn Sontesadisai hw
