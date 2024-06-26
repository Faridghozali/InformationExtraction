import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Fungsi untuk memuat dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Download stopwords dan punkt untuk tokenization
nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords bahasa Inggris
stop_words = set(stopwords.words('english'))

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Menghapus tanda baca dan karakter khusus
    tokens = [word for word in tokens if word.isalnum()]
    
    # Menghapus stopwords
    tokens = [word for word in tokens if not word in stop_words]
    
    # Menggabungkan kembali tokens menjadi kalimat
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

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
    selected_tab = st.radio('Pilih Tab:', ('Ekstraksi ', 'Data dan Penjelasan'))

# Judul aplikasi
st.title('Ekstraksi Pola Ujaran Kebencian')

# Sidebar dengan tab tambahan
with st.sidebar:
    st.subheader('Menu')
    selected_tab = st.radio('Pilih Tab:', ('Ekstraksi N-gram', 'Data dan Penjelasan'))

# Konten utama berdasarkan tab yang dipilih
if selected_tab == 'Ekstraksi N-gram':
    # Input teks dari pengguna
    user_input = st.text_area("Masukkan teks yang ingin dianalisis:", "")

    if user_input:
        # Preprocessing teks
        preprocessed_text = preprocess_text(user_input)
        
        # Pisahkan input yang sudah diproses menjadi kalimat-kalimat
        texts = preprocessed_text.split('\n')

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

elif selected_tab == 'Data dan Penjelasan':
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
    ### Penjelasan Dataset
    Anda dapat menambahkan penjelasan tambahan tentang dataset ini di sini.
    """)
