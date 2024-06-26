import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Fungsi untuk memuat dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Fungsi untuk ekstraksi n-gram
def extract_ngrams(texts, ngram_range=(1, 2)):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    X = vectorizer.fit_transform(texts)
    ngrams = vectorizer.get_feature_names_out()
    counts = X.toarray().sum(axis=0)
    df_ngrams = pd.DataFrame({'ngram': ngrams, 'count': counts})
    df_ngrams = df_ngrams.sort_values(by='count', ascending=False)
    return df_ngrams

# Fungsi untuk menghasilkan WordCloud
def generate_wordcloud(texts):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texts)
    return wordcloud

# Judul aplikasi
st.title('Ekstraksi Pola Ujaran Kebencian')

# Sidebar dengan tab tambahan
with st.sidebar:
    st.subheader('Menu')
    # Pilihan tab
    selected_tab = st.radio('Pilih Tab:', ('Data dan Penjelasan', 'Ekstraksi N-gram'))

# Konten utama berdasarkan tab yang dipilih
if selected_tab == 'Data dan Penjelasan':
    st.subheader('Data dan Penjelasan')
    st.markdown("""
    Di tab ini, Anda dapat menampilkan semua data yang relevan dan penjelasan terkait analisis atau hasil dari ekstraksi pola ujaran kebencian.
    
    ### Tabel Dataset: DATASET CYBERBULLYING INSTAGRAM - FINAL
    """)
    
    # Memuat dataset
    df_dataset = load_data('DATASET CYBERBULLYING INSTAGRAM - FINAL.csv')

    # Menampilkan tabel dataset
    st.dataframe(df_dataset)
    
    st.markdown("""
    ### WordCloud Berdasarkan Kategori
    """)

    # Memastikan kolom 'kategori' ada dalam dataset
    if 'Kategori' in df_dataset.columns:
        # Menggabungkan teks berdasarkan kategori
        bullying_texts = ' '.join(df_dataset[df_dataset['kategori'] == 'bullying']['text'].astype(str).tolist())
        non_bullying_texts = ' '.join(df_dataset[df_dataset['kategori'] == 'non bullying']['text'].astype(str).tolist())

        # Dropdown untuk memilih kategori
        category = st.selectbox('Pilih Kategori:', ['bullying', 'non bullying'])

        if category == 'bullying':
            # Membuat dan menampilkan WordCloud untuk teks bullying
            st.subheader('WordCloud Kategori Bullying')
            wordcloud_bullying = generate_wordcloud(bullying_texts)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_bullying, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

        elif category == 'non bullying':
            # Membuat dan menampilkan WordCloud untuk teks non-bullying
            st.subheader('WordCloud Kategori Non-Bullying')
            wordcloud_non_bullying = generate_wordcloud(non_bullying_texts)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_non_bullying, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
    else:
        st.error("Kolom 'kategori' tidak ditemukan dalam dataset. Pastikan nama kolom sesuai dengan struktur dataset.")

elif selected_tab == 'Ekstraksi N-gram':
    st.subheader('Ekstraksi N-gram')
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
        st.subheader('WordCloud N-gram')
        wordcloud = generate_wordcloud(user_input)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
