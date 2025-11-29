#Sentiment Analysis using NLP & Machine Learning

This project focuses on classifying sentiment (e.g., Positive, Negative, Neutral) from text data such as Twitter posts or user reviews using Natural Language Processing (NLP) techniques like text cleaning, TF-IDF vectorization, and a Naive Bayes classifier.

🚀 Project Overview

The goal of this project is to build a machine learning model that can automatically determine the sentiment of a given text.
This includes:

Cleaning raw text (removing URLs, punctuation, stopwords, etc.)

Converting text into numerical features using TF-IDF

Training a Multinomial Naive Bayes classifier

Evaluating model performance with accuracy, classification reports, and confusion matrix

Testing the model with custom user input

📁 Dataset

The project uses a CSV file named sentiment_dataset.csv with the following columns:

Column	Description
text	The actual sentence/tweet/review
sentiment	The label (e.g., positive/neutral/negative)

Make sure this file is placed in the same directory as the script.

🧠 Techniques Used
NLP Techniques

Text lowercasing

URL & mention removal

Removing punctuation

Stopwords filtering

Lemmatization

TF-IDF Vectorization

Machine Learning

Multinomial Naive Bayes classifier

Train-Test split

Model evaluation metrics

🛠️ Dependencies

Install all required libraries using:

pip install pandas numpy nltk scikit-learn


The script will also download NLTK resources automatically:

stopwords

wordnet

📘 How the Code Works
1. Load dataset
df = pd.read_csv("sentiment_dataset.csv")

2. Clean text

A custom clean_text() function performs:

lowercasing

regex cleaning

punctuation removal

lemmatization

stopword removal

3. Split dataset
train_test_split(..., test_size=0.2)

4. Feature extraction
TfidfVectorizer(max_features=5000)

5. Train Naive Bayes classifier
model.fit(X_train_tfidf, y_train)

6. Evaluate

Accuracy

Classification Report

Confusion Matrix

7. Predict sentiment for a new input
model.predict(...)

📊 Model Evaluation Example

The script prints:

✔️ Accuracy

✔️ Precision/Recall/F1-score

✔️ Confusion Matrix

▶️ How to Run

Clone/download the project

Install dependencies

Place sentiment_dataset.csv in the folder

Run:

python sentiment_analysis.py

💬 Example Output
Accuracy: 0.87

Predicted Sentiment: positive

🔮 Future Improvements

Try other models (SVM, Logistic Regression, Random Forest)

Use deep learning (LSTM, BERT, RoBERTa)

Add hyperparameter tuning (GridSearchCV)

Create a Streamlit web app for live predictions

📄 License

This project is open-source and free to use.
