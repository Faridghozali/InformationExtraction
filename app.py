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

# Judul aplikasi
st.title('Ekstraksi Pola Ujaran Kebencian')

# Sidebar dengan tab tambahan
with st.sidebar:
    st.subheader('Menu')
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

    # Periksa nama kolom yang ada dalam DataFrame
    st.write(df_dataset.columns)

    # Menghitung jumlah label 'bullying' dan 'non-bullying' jika nama kolom benar
    if 'label' in df_dataset.columns:
        bullying_count = df_dataset[df_dataset['label'] == 'bullying'].shape[0]
        non_bullying_count = df_dataset[df_dataset['label'] == 'non bullying'].shape[0]

        st.write(f"Jumlah 'bullying': {bullying_count}")
        st.write(f"Jumlah 'non-bullying': {non_bullying_count}")

        # Visualisasi jumlah label
        st.subheader('Visualisasi Jumlah Label')
        plt.figure(figsize=(8, 5))
        plt.bar(['bullying', 'non-bullying'], [bullying_count, non_bullying_count])
        plt.xlabel('Label')
        plt.ylabel('Jumlah')
        st.pyplot(plt)
    else:
        st.write("Kolom 'label' tidak ditemukan dalam dataset. Periksa kembali nama kolomnya.")

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
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(df_ngrams.set_index('ngram').to_dict()['count'])

        st.subheader('WordCloud N-gram')
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
