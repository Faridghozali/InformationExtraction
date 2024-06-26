import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Fungsi untuk ekstraksi n-gram
def extract_ngrams(texts, ngram_range=(1, 2)):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    X = vectorizer.fit_transform(texts)
    ngrams = vectorizer.get_feature_names_out()
    counts = X.toarray().sum(axis=0)
    df_ngrams = pd.DataFrame({'ngram': ngrams, 'count': counts})
    df_ngrams = df_ngrams.sort_values(by='count', ascending=False)
    return df_ngrams

# Judul aplikasi
st.title('Ekstraksi Pola Ujaran Kebencian')

# Input teks dari pengguna
user_input = st.text_area("Masukkan teks yang ingin dianalisis:", "")

if user_input:
    # Pisahkan input menjadi kalimat-kalimat
    texts = user_input.split('\n')
    
    # Ekstraksi n-gram
    df_ngrams = extract_ngrams(texts)
    
    # Tampilkan tabel n-gram dan frekuensinya
    st.subheader('Frekuensi N-gram')
    st.dataframe(df_ngrams)
    
    # Visualisasi WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(df_ngrams.set_index('ngram').to_dict()['count'])
    
    st.subheader('WordCloud N-gram')
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

