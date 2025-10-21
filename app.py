import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

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

# Session state başlat (chat history için)
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_started' not in st.session_state:
    st.session_state.chat_started = False

# API Key kontrolü
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    # Önce environment'tan kontrol et
    api_key = os.environ.get("GEMINI_API_KEY", "")
    
    if api_key:
        st.success("✅ API Key yüklendi!")
    else:
        api_key = st.text_input("Gemini API Key", type="password", help="Ücretsiz key almak için: https://aistudio.google.com/apikey")
        if not api_key:
            st.warning("⚠️ Lütfen API Key giriniz.")
    
    st.markdown("---")
    st.markdown("### 📊 Proje Bilgileri")
    st.info("""
    - **Dataset:** 42,804 Türkçe tıbbi makale
    - **Chunk:** 244,150 parça
    - **Model:** Gemini 2.5 Flash
    - **Embedding:** Multilingual MiniLM
    - **Vector DB:** FAISS
    """)
    
    if st.button("🗑️ Sohbeti Temizle"):
        st.session_state.messages = []
        st.session_state.chat_started = False
        st.rerun()

# API key yoksa dur
if not api_key:
    st.info("👈 Lütfen soldaki menüden API Key giriniz.")
    st.stop()

# Model ve verileri yükle (cache ile)
@st.cache_resource
def load_models_and_data():
    # Embedding model
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # FAISS index ve chunks yükle
    with open('faiss_index.pkl', 'rb') as f:
        data = pickle.load(f)
    
    index = data['index']
    chunks = data['chunks']
    metadatas = data['metadatas']
    
    return embedding_model, index, chunks, metadatas

try:
    with st.spinner("🔄 Model yükleniyor..."):
        embedding_model, index, chunks, metadatas = load_models_and_data()
    
    # Gemini yapılandır
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    if not st.session_state.chat_started:
        st.success("✅ Sistem hazır! Soru sorabilirsiniz.")
        st.session_state.chat_started = True
    
except Exception as e:
    st.error(f"❌ Model yüklenirken hata: {e}")
    st.stop()

# RAG fonksiyonu (CHATBOT tarzı)
def rag_query(question, top_k=3):
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
    context = "\n\n".join([
        f"[{title}]: {chunk[:300]}..." 
        for title, chunk in zip(retrieved_titles, retrieved_chunks)
    ])
    
    # Chatbot tarzı prompt
    prompt = f"""Sen Türkçe konuşan, samimi ve yardımsever bir sağlık asistanısın. 

KAYNAKLARDAN BİLGİLER:
{context}

KULLANICI SORUSU: {question}

ÖNEMLİ KURALLAR:
- Kısa, özlü ve konuşma tarzında cevap ver (maksimum 3-4 cümle)
- Gereksiz detaya girme, doğrudan cevapla
- Samimi ol ama profesyonel kal
- Kaynaklarda bilgi yoksa "Bu konuda kaynaklarda yeterli bilgi bulamadım, lütfen bir sağlık profesyoneline danışın" de
- Cevabın sonunda kaynak başlıklarını belirtme (sistem otomatik gösterecek)

CEVAP:"""

    # Gemini'den cevap al
    response = model.generate_content(prompt)
    answer = response.text
    
    unique_sources = list(set(retrieved_titles))[:3]
    
    return answer, unique_sources

# Chat arayüzü
st.markdown("### 💬 Sohbet")

# Önceki mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 Kaynaklar"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"{i}. {source}")

# Yeni mesaj girişi
if prompt := st.chat_input("Sağlık hakkında bir soru sorun..."):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Bot cevabı
    with st.chat_message("assistant"):
        with st.spinner("🤔 Düşünüyorum..."):
            try:
                answer, sources = rag_query(prompt, top_k=3)
                st.markdown(answer)
                
                with st.expander("📚 Kaynaklar"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"{i}. {source}")
                
                # Mesajı kaydet
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"❌ Hata oluştu: {e}")

# Örnek sorular (ilk başta göster)
if len(st.session_state.messages) == 0:
    st.markdown("---")
    st.markdown("### 💡 Örnek Sorular:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.button("🩺 Diyabet nedir?", key="ex1", use_container_width=True)
    with col2:
        st.button("💊 Migren nasıl geçer?", key="ex2", use_container_width=True)
    with col3:
        st.button("🤰 Hamilelik testleri", key="ex3", use_container_width=True)
