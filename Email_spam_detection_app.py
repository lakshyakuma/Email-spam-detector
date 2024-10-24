import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_Message(Message):
    Message = Message.lower()                               # Lower case
    Message = nltk.word_tokenize(Message)               # Tokenization
    
    list_word = []
                                                            # Removing special character
    for i in Message:
        if i.isalnum():
            list_word.append(i)

    Message = list_word[:]                                   # As list cannot be directly assign , we need to cloning as its a mutable datatype
    list_word.clear()

    for i in Message:
        if i not in stopwords.words('english') and i not in string.punctuation:             # Removing stopwords and punctuation 
            list_word.append(i)
            
    Message = list_word[:]                                   # As list cannot be directly assign , we need to cloning as its a mutable datatype
    list_word.clear()

    for i in Message:
        list_word.append(ps.stem(i))
        
    
    return list_word



# Define the paths for your model and vectorizer

model_path = os.path.join('models', 'spam_detector_model.pkl')
vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')

# Load the model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the vectorizer
with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)

# Title of the app
st.title("Email Spam Detector")

# Input text from user
user_input_mail = st.text_area("Enter the email content to check if it's spam or not:")
#    1. Preprocess
transformed_mail = ' '.join(transform_Message(user_input_mail))  # Join list into string

if st.button('Predict'):
#        2. Vectorize by using vectorizer
    vector_input = vectorizer.transform([transformed_mail])
#        3. Predict using the model
    prediction_result = model.predict(vector_input)[0]
#        4. Display the model result
    if prediction_result == 1:

        st.write("This email is **Spam**.")
    else:

        st.write("This email is **Not Spam**.")
    






