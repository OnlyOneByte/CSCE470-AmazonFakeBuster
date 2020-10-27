# CSCE 470 - Amazon Fake Buster
## Requirements
- Anaconda/Miniconda on Python 3
  - pandas
  - numpy
  - scipy
  - sklearn
  - scikit-learn
  - jupyter notebook
- nltk
  - either `conda install nltk` or `pip install nltk`

## Data Exploration
The `data_exploration.ipynb` notebook explores the content of the data source, and checks to see various specifics of the dataset. This was used in determining the best course of action in regard to the core algorithm, as well as in general. We used our intuition in this file to be able to determine which of the items worked the best.

## Model Generation
The `core_algo.ipynb` notebook generates and pickles the SVM classifier we trained and tuned to classify the corpus. 

## Running the classifier
Data is expected to be a vector of:
- Rating: integer [1-5]
- Product Category, string
- Verified Purchase, string either `Y` or `N`
- The parsed review text
  - Tokenized
  - Lemmatized
  - stop works filtered
  - and bigrams created.
In that order.

### Example  
An example demo of:
1. Loading the pickle
2. Parsing the test
3. Prediction on the text

is given in `example_usage.py`. THis can be ran by `python example_usage.py`

