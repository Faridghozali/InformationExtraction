import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Function to load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Function to extract n-grams
def extract_ngrams(texts, ngram_range=(1, 2)):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    X = vectorizer.fit_transform(texts)
    ngrams = vectorizer.get_feature_names_out()
    counts = X.toarray().sum(axis=0)
    df_ngrams = pd.DataFrame({'ngram': ngrams, 'count': counts})
    df_ngrams = df_ngrams.sort_values(by='count', ascending=False)
    return df_ngrams

# Function to generate WordCloud
def generate_wordcloud(texts):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texts)
    return wordcloud

# Title of the application
st.title('Ekstraksi Pola Ujaran Kebencian')

# Sidebar with tab options
with st.sidebar:
    st.subheader('Menu')
    selected_tab = st.radio('Pilih Tab:', ('Data dan Penjelasan', 'Ekstraksi N-gram'))

# Main content based on selected tab
if selected_tab == 'Data dan Penjelasan':
    st.subheader('Data dan Penjelasan')
    st.markdown("""
    Di tab ini, Anda dapat menampilkan semua data yang relevan dan penjelasan terkait analisis atau hasil dari ekstraksi pola ujaran kebencian.
    
    ### Tabel Dataset: DATASET CYBERBULLYING INSTAGRAM - FINAL
    """)

    # Load dataset
    df_dataset = load_data('DATASET CYBERBULLYING INSTAGRAM - FINAL.csv')

    # Display dataset table
    st.dataframe(df_dataset)

    st.markdown("""
    ### WordCloud Kalimat Bullying dan Non-Bullying
    """)

    # Ensure 'label' column exists and is correctly spelled
    if 'label' in df_dataset.columns:
        # Filter data based on labels
        bullying_texts = ' '.join(df_dataset[df_dataset['label'] == 'bullying']['text'].astype(str).tolist())
        non_bullying_texts = ' '.join(df_dataset[df_dataset['label'] == 'non bullying']['text'].astype(str).tolist())

        # Dropdown to select category
        category = st.selectbox('Pilih Kategori:', ['bullying', 'non bullying'])

        if category == 'bullying':
            # Generate and display WordCloud for bullying texts
            st.subheader('WordCloud Kalimat Bullying')
            wordcloud_bullying = generate_wordcloud(bullying_texts)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_bullying, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

        elif category == 'non bullying':
            # Generate and display WordCloud for non-bullying texts
            st.subheader('WordCloud Kalimat Non-Bullying')
            wordcloud_non_bullying = generate_wordcloud(non_bullying_texts)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_non_bullying, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
    else:
        st.error("Kolom 'label' tidak ditemukan dalam dataset. Pastikan nama kolom sesuai dengan struktur dataset.")

elif selected_tab == 'Ekstraksi N-gram':
    st.subheader('Ekstraksi N-gram')
    # Input text from user
    user_input = st.text_area("Masukkan teks yang ingin dianalisis:", "")

    if user_input:
        # Split input into sentences
        texts = user_input.split('\n')

        # Extract n-grams
        df_ngrams = extract_ngrams(texts)

        # Display n-gram frequencies as a table
        st.subheader('Frekuensi N-gram')
        st.dataframe(df_ngrams)

        # Generate and display WordCloud for input text
        st.subheader('WordCloud N-gram')
        wordcloud = generate_wordcloud(user_input)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
