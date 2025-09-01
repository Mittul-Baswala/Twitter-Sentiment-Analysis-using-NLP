# 💬 Twitter Sentiment Analysis using NLP

*COMPANY NAME* : CODTECH IT SOLUTIONS 

*NAME*: MITTUL BASWALA

*INTERN ID*: CT06DZ2256

*DOMAIN*: DATA ANALYTICS

*MENTOR*: NEELA SANTOSH


## 📌 Overview
This project focuses on analyzing sentiments from tweets using Natural Language Processing (NLP) techniques and Machine Learning models. The main objective is to classify tweets into sentiment categories such as Positive, Negative, and Neutral, providing insights into public opinion and trends.

## 📂 Dataset
- Source: [Twitter Sentiment Dataset]( https://www.kaggle.com/datasets/prakashpraba/twitter-sentiment-analysis-dataset)
- Columns:
  - tweet → Original tweet text
  - label → Sentiment label (0 = Negative, 1 = Neutral, 2 = Positive)
- Data Cleaning & Preprocessing:
  - Removal of URLs, mentions (@user), hashtags, numbers, and special characters
  - Conversion to lowercase
  - Stopwords removal using NLTK
  - Lemmatization for reducing words to base form

## 🧼 Preprocessing
- Removed URLs, mentions, hashtags, and special characters
- Converted text to lowercase
- Removed stopwords using NLTK
- Tokenized and cleaned tweets for modeling


## ⚙️ Technologies & Libraries
- Python 3.11
- Libraries Used:
  - Data Handling: pandas, numpy
  - NLP: nltk, re
  - Visualization: matplotlib, seaborn
  - Machine Learning: scikit-learn

## 🛠️ Methodology
- Data Preprocessing
  - Cleaning tweets and normalizing text.
  - Generating a new column cleaned_tweet.
- Exploratory Data Analysis (EDA)
  - Distribution of sentiment classes.
  - Word frequency insights.
  - Visualizations using bar charts & count plots.
- Feature Extraction
  - Applied TF-IDF Vectorization to convert text into numerical features.
  - Limited features to top 5,000 most informative tokens.
- Model Building
  - Implemented Logistic Regression using a pipeline with StandardScaler.
  - Split dataset into train/test sets (80/20).
- Model Evaluation
  - Metrics: Accuracy, Precision, Recall, F1-score.
  - Confusion Matrix heatmap to visualize predictions.

## 📊 Key Insights
- Balanced dataset between sentiment classes (slight skew possible).
- Logistic Regression performed well with high accuracy.
- Misclassifications occurred mostly between Neutral vs Positive tweets.
- TF-IDF captured relevant patterns in language for classification.

## 🚀 Results
- Accuracy: ~80-85% (depending on dataset split).
- Classification Report:
  - Precision and Recall scores show strong performance for Positive and Negative tweets.
  - Neutral tweets had slightly lower recall due to overlapping language patterns.

- Visualization Outputs:
  - Count plots of sentiment distribution.
  - Confusion matrix highlighting prediction strengths/weaknesses.

## ✅ Deliverables
- Jupyter Notebook with full pipeline
- README.md (this file)
- Visuals and charts
