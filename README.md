# 🎬 IMDB Movie Reviews Sentiment Analysis

This project classifies IMDB movie reviews as **positive** or **negative** using NLP and Machine Learning.

## 📊 Dataset
- Dataset: [IMDB Dataset of 50K Movie Reviews (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Balanced dataset: 25,000 positive, 25,000 negative reviews

## 🛠️ Features
- Text cleaning (HTML stripping, stopword removal, stemming)
- Bag of Words, TF-IDF, and Word2Vec representations
- Multiple models: Logistic Regression, Naive Bayes, Random Forest, SVM, SGDClassifier
- WordCloud visualization for frequent positive & negative words
- Trained Logistic Regression model saved as `lr.pkl`

## 🚀 How to Run
Clone repo:
```bash```
git clone https://github.com/your-username/IMDB-Sentiment-Analysis.git
cd IMDB-Sentiment-Analysis
## Install dependencies:

pip install -r requirements.txt


## Run notebook:

jupyter notebook notebooks/IMDBSentimentAnalysis.ipynb

🧪 Predict New Reviews
import pickle
from text_cleaning import denoise_text

lr = pickle.load(open("models/lr.pkl","rb"))
tf = pickle.load(open("models/tf.pkl","rb"))

def predict(text):
    cleaned = denoise_text(text)
    tf_text = tf.transform([cleaned])
    return lr.predict(tf_text)[0]

print(predict("This movie was amazing!"))


Output:

0 → Negative

1 → Positive
