import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Fungsi untuk memuat dataset
@st.cache
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

    # Menggabungkan semua teks dalam satu variabel
    all_texts = ' '.join(df_dataset['text'].astype(str).tolist())

    # Ekstraksi n-gram dari teks
    df_ngrams = extract_ngrams([all_texts])

    # Visualisasi WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(df_ngrams.set_index('ngram').to_dict()['count'])

    st.subheader('WordCloud N-gram dari Dataset')
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

elif selected_tab == 'Ekstraksi N-gram':
    st.subheader('Ekstraksi N-gram')
    user_input = st.text_area("Masukkan teks yang ingin dianalisis:", "")
    
    if user_input:
        texts = user_input.split('\n')
        df_ngrams = extract_ngrams(texts)
        
        st.subheader('Frekuensi N-gram')
        st.dataframe(df_ngrams)
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(df_ngrams.set_index('ngram').to_dict()['count'])
        
        st.subheader('WordCloud N-gram')
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
