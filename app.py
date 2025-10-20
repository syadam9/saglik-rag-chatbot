
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Sayfa ayarları
st.set_page_config(
    page_title="Türkçe Sağlık Asistanı",
    page_icon="🏥",
    layout="wide"
)

# Başlık
st.title("🏥 Türkçe Sağlık Bilgi Asistanı")
st.markdown("**43,000+ tıbbi makaleden bilgi çeken RAG tabanlı chatbot**")
st.markdown("---")

# API Key girişi (sidebar)
with st.sidebar:
    st.header("⚙️ Ayarlar")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("---")
    st.markdown("### 📊 Proje Bilgileri")
    st.info("""
    - **Dataset:** 42,804 Türkçe tıbbi makale
    - **Chunk:** 244,150 parça
    - **Model:** Gemini 2.5 Flash
    - **Embedding:** Multilingual MiniLM
    - **Vector DB:** FAISS
    """)

# Ana içerik
if not api_key:
    st.warning("⚠️ Lütfen soldaki menüden Gemini API Key giriniz.")
    st.stop()

# Model ve verileri yükle (cache ile)
@st.cache_resource
def load_models_and_data():
    # Embedding model
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # FAISS index ve chunks yükle
    with open('faiss_index.pkl', 'rb') as f:
        data = pickle.load(f)
    
    index = data['index']
    chunks = data['chunks']
    metadatas = data['metadatas']
    
    return embedding_model, index, chunks, metadatas

try:
    embedding_model, index, chunks, metadatas = load_models_and_data()
    
    # Gemini yapılandır
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    st.success("✅ Sistem hazır! Soru sorabilirsiniz.")
    
except Exception as e:
    st.error(f"❌ Model yüklenirken hata: {e}")
    st.stop()

# Soru-cevap fonksiyonu
def rag_query(question, top_k=5):
    # Soruyu embedding'e çevir
    question_embedding = embedding_model.encode([question])[0]
    question_embedding = np.array([question_embedding]).astype('float32')
    
    # FAISS'te ara
    distances, indices = index.search(question_embedding, top_k)
    
    # Chunk'ları al
    retrieved_chunks = []
    retrieved_titles = []
    
    for idx in indices[0]:
        retrieved_chunks.append(chunks[idx])
        retrieved_titles.append(metadatas[idx]['title'])
    
    # Context oluştur
    context = "\n\n---\n\n".join([
        f"Kaynak: {title}\n{chunk}" 
        for title, chunk in zip(retrieved_titles, retrieved_chunks)
    ])
    
    # Prompt
    prompt = f"""Sen Türkçe tıbbi bir asistansın. Aşağıdaki tıbbi makalelerden yararlanarak soruyu cevapla.

KAYNAKLARDAN ALINAN BİLGİLER:
{context}

SORU: {question}

CEVAP: Yukarıdaki kaynaklara dayanarak, soruyu açık ve anlaşılır bir şekilde Türkçe olarak cevaplayın."""

    # Gemini'den cevap al
    response = model.generate_content(prompt)
    answer = response.text
    
    unique_sources = list(set(retrieved_titles))
    
    return answer, unique_sources

# Soru girişi
st.markdown("### 💬 Sorunuzu Sorun")

col1, col2 = st.columns([4, 1])

with col1:
    question = st.text_input("", placeholder="Örn: Diyabet nedir ve belirtileri nelerdir?")

with col2:
    top_k = st.selectbox("Kaynak sayısı", [3, 5, 7], index=1)

if st.button("🔍 Sorgula", type="primary"):
    if question:
        with st.spinner("🤔 Cevap hazırlanıyor..."):
            try:
                answer, sources = rag_query(question, top_k=top_k)
                
                st.markdown("### 🤖 Cevap")
                st.markdown(answer)
                
                st.markdown("---")
                st.markdown("### 📚 Kullanılan Kaynaklar")
                for i, source in enumerate(sources[:5], 1):
                    st.markdown(f"{i}. {source}")
                    
            except Exception as e:
                st.error(f"❌ Hata oluştu: {e}")
    else:
        st.warning("⚠️ Lütfen bir soru girin.")

# Örnek sorular
st.markdown("---")
st.markdown("### 💡 Örnek Sorular")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Diyabet nedir?"):
        st.session_state.example_q = "Diyabet nedir ve belirtileri nelerdir?"

with col2:
    if st.button("Migren nasıl geçer?"):
        st.session_state.example_q = "Migren ağrısı nasıl geçer?"

with col3:
    if st.button("Hamilelik testleri"):
        st.session_state.example_q = "Hamilelerde yapılan testler nelerdir?"
