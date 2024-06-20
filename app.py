import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from spacy import displacy
import gensim
import pyLDAvis.gensim as gensimvis

# Function to clean text
def clean_text(text):
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    # Remove URLs, mentions, and hashtags from the text
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    # Tokenize the text
    words = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    words = [w for w in words if w not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    # Join the words back into a string
    text = ' '.join(words)
    return text

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv(DATASET CYBERBULLYING INSTAGRAM - FINAL.csv')
    return df

# Main function to run the app
def main():
    st.title('Aplikasi Deteksi Cyberbullying di Komentar Instagram')
    
    # Load data
    df = load_data()
    
    # Sidebar options
    activity = st.sidebar.selectbox("Pilih aktivitas", ["Tampilkan Data", "Visualisasi", "Model Prediksi"])
    
    if activity == "Tampilkan Data":
        st.subheader("Tampilkan Data")
        st.write(df.head())
        
    elif activity == "Visualisasi":
        st.subheader("Visualisasi Data")
        # Visualize class distribution
        st.subheader('Distribusi Kelas dari Kategori CyberBullying')
        fig = plt.figure()
        sns.countplot(x='Kategori', data=df)
        st.pyplot(fig)
        
        # Word cloud
        st.subheader('Word Cloud dari Komentar')
        text = ' '.join(df['Komentar'])
        wordcloud = WordCloud(width=800, height=400).generate(text)
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot()
        
        # Named Entity Recognition (NER)
        st.subheader('Contoh Named Entity Recognition (NER)')
        text = df['Komentar'].iloc[0]
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        html = displacy.render(doc, style='ent')
        st.write(html, unsafe_allow_html=True)
        
    elif activity == "Model Prediksi":
        st.subheader("Prediksi Kategori Cyberbullying")
        
        # Preprocess text data
        df['cleaned_text'] = df['Komentar'].apply(clean_text)
        
        # Create Bag of Words model
        cv = CountVectorizer()
        X = cv.fit_transform(df['cleaned_text']).toarray()
        y = df['Kategori']
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a Logistic Regression model
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy: {accuracy:.2f}')
        
        # Confusion matrix
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
        
        # Classification report
        st.subheader('Classification Report')
        report = classification_report(y_test, y_pred)
        st.write(report)

if __name__ == '__main__':
    main()
