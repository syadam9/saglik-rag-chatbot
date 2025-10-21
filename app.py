import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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

# Session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    
    if not api_key:
        api_key = st.text_input("Gemini API Key", type="password")
        if not api_key:
            st.warning("⚠️ Lütfen API Key giriniz.")
    
    if st.button("🗑️ Sohbeti Temizle"):
        st.session_state.messages = []
        st.rerun()

# API key kontrol
if not api_key:
    st.info("👈 Lütfen soldaki menüden API Key giriniz.")
    st.stop()

# Model ve verileri yükle
@st.cache_resource
def load_data():
    # Embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Basit veri yükleme (FAISS yerine)
    try:
        with open('medical_chunks.pkl', 'rb') as f:
            data = pickle.load(f)
        chunks = data['chunks']
        embeddings = data['embeddings']
        titles = data['titles']
    except:
        # Fallback: basit örnek veri
        chunks = ["Baş ağrısı stres ve yorgunluktan kaynaklanabilir.", 
                 "Migren genellikle zonklayıcı baş ağrısı şeklinde görülür."]
        embeddings = model.encode(chunks)
        titles = ["Baş Ağrısı", "Migren"]
    
    return model, chunks, embeddings, titles

try:
    with st.spinner("🔄 Sistem yükleniyor..."):
        model, chunks, embeddings, titles = load_data()
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    st.success("✅ Sistem hazır! Soru sorabilirsiniz.")
except Exception as e:
    st.error(f"❌ Yükleme hatası: {e}")
    st.stop()

# Basit RAG fonksiyonu
def simple_rag(question, top_k=3):
    # Soruyu encode et
    question_embedding = model.encode([question])
    
    # Cosine similarity ile benzerlik bul
    similarities = cosine_similarity(question_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Context oluştur
    context = ""
    for idx in top_indices:
        context += f"{titles[idx]}: {chunks[idx]}\n\n"
    
    # Prompt
    prompt = f"""Sen Türkçe konuşan, samimi bir sağlık asistanısın.

BİLGİLER:
{context}

SORU: {question}

KURALLAR:
- Kısa, özlü cevap ver (2-3 cümle)
- Samimi ve yardımsever ol
- Kaynak yoksa "Bu konuda yeterli bilgim yok" de

CEVAP:"""
    
    # Gemini'den cevap al
    try:
        response = gemini_model.generate_content(prompt)
        return response.text, [titles[i] for i in top_indices]
    except Exception as e:
        return f"Üzgünüm, bir hata oluştu: {e}", []

# Chat arayüzü
st.markdown("### 💬 Sohbet")

# Mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 Kaynaklar"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")

# Yeni mesaj
if prompt := st.chat_input("Sağlık hakkında soru sorun..."):
    # Kullanıcı mesajı
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Bot cevabı
    with st.chat_message("assistant"):
        with st.spinner("🤔 Düşünüyorum..."):
            answer, sources = simple_rag(prompt)
            st.markdown(answer)
            
            if sources:
                with st.expander("📚 Kaynaklar"):
                    for source in sources:
                        st.markdown(f"- {source}")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })

# Örnek sorular
if len(st.session_state.messages) == 0:
    st.markdown("### 💡 Örnek Sorular:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🩺 Baş ağrısı neden olur?"):
            st.rerun()
    with col2:
        if st.button("💊 Migren belirtileri neler?"):
            st.rerun()
