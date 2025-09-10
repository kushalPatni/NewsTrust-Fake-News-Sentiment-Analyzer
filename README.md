# NewsTrust-Fake-News-Sentiment-Analyzer
📌 Overview

NewsTrust is a Machine Learning & Natural Language Processing (NLP) project that combines Fake News Detection and Sentiment Analysis for news headlines.

The project aims to:

Identify whether a news headline is Fake or True.

Analyze the sentiment of the headline (Positive, Neutral, Negative).

Provide an interactive Streamlit UI where users can input a news headline and instantly get predictions along with confidence scores.

This project was developed as part of an academic submission but is also designed to be extendable for real-world use cases.

🎯 Objectives

✅ Detect fake vs. authentic news with high accuracy.

✅ Analyze the emotional tone (sentiment) of news headlines.

✅ Build an interactive dashboard for real-time predictions.

✅ Provide dataset visualizations and explain model insights.

📂 Dataset
1. Fake News Detection

Dataset: Kaggle Fake & True News dataset

Samples: ~44,000 news articles

Classes: Fake (0), True (1)

Balanced dataset to improve classification.

2. Sentiment Analysis

Datasets Used:

CNBC Sentiment Dataset

Guardian Sentiment Dataset

Reuters Sentiment Dataset

Merged Size: ~53,000 headlines

Labels: Sentiment scores (1–5) remapped into 3 classes:

0 → Negative (scores 1 & 2)

1 → Neutral (score 3)

2 → Positive (scores 4 & 5)

🛠️ Data Preprocessing

For both fake news and sentiment datasets:

Convert text to lowercase

Remove URLs, emails, numbers, punctuation

Remove stopwords

Tokenization using NLTK

Lemmatization to reduce words to root form

Vectorization:

Fake News → Tokenizer + BERT embeddings

Sentiment → TF-IDF (1-3 n-grams, 30k features)

🤖 Models Used
🔍 Fake News Detection

Architecture: Custom BERT + Fully Connected Layers

Training: Fine-tuned on Kaggle dataset

Performance:

Accuracy: 89%

Precision/Recall balanced across Fake & True

😊 Sentiment Analysis

Models Tested: Logistic Regression, Random Forest, SVM

Best Model: Random Forest with TF-IDF

Performance:

Accuracy: ~70%

Neutral class underperforming due to imbalance

📊 Results
| Metric    | Fake | True | Average  |
| --------- | ---- | ---- | -------- |
| Precision | 0.87 | 0.90 | 0.89     |
| Recall    | 0.89 | 0.88 | 0.89     |
| F1-score  | 0.88 | 0.89 | 0.89     |
| Accuracy  |      |      | **0.89** |


| Sentiment            | Precision | Recall | F1-score |
| -------------------- | --------- | ------ | -------- |
| Negative             | 0.75      | 0.84   | 0.79     |
| Neutral              | 0.40      | 0.13   | 0.20     |
| Positive             | 0.61      | 0.58   | 0.60     |
| **Overall Accuracy** |           |        | **0.70** |

🖥️ Streamlit UI

The project includes a Streamlit app with three main sections:

📊 Dataset Insights

Visualizations: Sentiment distribution, Fake vs True split

Word clouds for each sentiment class

🔮 Analyze Headlines

Input box for entering a news headline

Predictions: Fake/True + Sentiment with confidence scores

Visualization of sentiment probabilities

ℹ️ About Section

Project details

Dataset info

Limitations & future improvements

⚙️ Installation

Clone the repo

git clone https://github.com/your-username/NewsTrust.git
cd NewsTrust


Create a virtual environment

conda create -n newstrust python=3.10
conda activate newstrust


Install dependencies

pip install -r requirements.txt


Run Streamlit app

streamlit run ui.py

📦 Project Structure
NewsTrust/
│── data/                      # Datasets (Kaggle + merged sentiment dataset)
│── models/                    # Saved models (fake_news_bert.pth, sentiment_model.pkl, tfidf_vectorizer.pkl)
│── notebooks/                 # Jupyter notebooks for training
│── ui.py                      # Streamlit user interface
│── requirements.txt           # Dependencies
│── README.md                  # Project documentation

🚀 Future Improvements

Handle sentiment imbalance (especially Neutral class)

Multi-language news support

Integration with fact-checking APIs

Deploy on cloud for real-time usage

🙌 Acknowledgements

Datasets: Kaggle, CNBC, Guardian, Reuters

Libraries: PyTorch, HuggingFace Transformers, Scikit-learn, Streamlit, NLTK

Inspired by the need to fight misinformation and promote media literacy.

🔥 With NewsTrust, users can instantly verify if a news headline is fake or true, and also understand its emotional impact.
