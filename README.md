# Email-spam-detector
This project is a Streamlit web application that detects whether an email is spam or not using a pre-trained machine learning model. The app takes email content as input, preprocesses it, and classifies it as either "Spam" or "Not Spam" using a Multinomial Naive Bayes model and TF-IDF Vectorizer.

Key Features:

Email Content Analysis: Input raw email text to check if it's spam.

Natural Language Processing: The app uses nltk for tokenization, stopword removal, and stemming.

Machine Learning Model: Built with Multinomial Naive Bayes trained on a spam detection dataset.

Streamlit Framework: Easy-to-use and interactive web application for real-time predictions.

Technologies Used:
Python: Core programming language.
Streamlit: For creating the web interface.
scikit-learn: For machine learning model building and text vectorization.
nltk: For text preprocessing (tokenization, stemming, stopwords).

How It Works:
The user inputs an email's content.
The app preprocesses the email (removes stopwords, punctuation, stems words, etc.).
The processed email is vectorized using a TF-IDF vectorizer.
The Naive Bayes model predicts whether the email is spam or not.
The result is displayed as either "Spam" or "Not Spam"
