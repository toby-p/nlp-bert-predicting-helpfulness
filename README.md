# w266-final-project
Final project for UCB MIDS W266 NLP class

## Data
The data sets used in this project can be found in the /data folder. We use review data sourced from Amazon and from Yelp.
/data
  /amazon
  /yelp
  
## EDA 
Initial EDA and creation of the Yelp data sets can be found in [Yelp_Data_EDA.ipynb](https://github.com/toby-p/nlp-bert-predicting-helpfulness/blob/main/Yelp_Data_EDA.ipynb)

Processing of the Amazon data can be found in [Amazon-data-processing-LARGE.ipynb](https://github.com/toby-p/nlp-bert-predicting-helpfulness/blob/main/Amazon-data-processing-LARGE.ipynb) and [Amazon-data-processing-SMALL.ipynb](https://github.com/toby-p/nlp-bert-predicting-helpfulness/blob/main/Amazon-data-processing-SMALL.ipynb)

## Baseline models
Recreation of the baseline model form Bilal et. al. is in [Bilal_et_al_Baseline.ipynb](https://github.com/toby-p/nlp-bert-predicting-helpfulness/blob/main/Bilal_et_al_Baseline.ipynb)

In [Bilal_et_al_baseline_on_Yelp_data.ipynb](https://github.com/toby-p/nlp-bert-predicting-helpfulness/blob/main/Bilal_et_al_baseline_on_Yelp_data.ipynb) we fine-tune the baseline model using the Yelp data set, to create a new baseline.

## Model Training
Fine-tuning of the RoBERTa model can be found in [RoBERTa.ipynb](https://github.com/toby-p/nlp-bert-predicting-helpfulness/blob/main/RoBERTa.ipynb)

Model training using transfer learning techniques can be found in [transfer_learning.py](https://github.com/toby-p/nlp-bert-predicting-helpfulness/blob/main/transfer_learning.py), [train_bilal_baseline.py](https://github.com/toby-p/nlp-bert-predicting-helpfulness/blob/main/train_bilal_baseline.py) and [train_amazon.py](https://github.com/toby-p/nlp-bert-predicting-helpfulness/blob/main/train_amazon.py), [train_amazon_large.py](https://github.com/toby-p/nlp-bert-predicting-helpfulness/blob/main/train_amazon_large.py)

The evalution of transfer learning can be found in [Evaluating Transfer Learning Models.ipynb](https://github.com/toby-p/nlp-bert-predicting-helpfulness/blob/main/Evaluating%20Transfer%20Learning%20Models.ipynb)

## Model checkpoints
Saved models can be found in /results, not that not all models are stored here due to size contraints.
