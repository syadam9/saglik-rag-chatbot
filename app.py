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
    layout="centered"
)

# CSS - Chat görünümü
st.markdown("""
<style>
    .user-msg {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .bot-msg {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Başlık
st.title("🏥 Türkçe Sağlık Asistanı")
st.caption("43,000+ tıbbi makaleden bilgi çeken AI chatbot")

# Session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    # API Key
    api_key = os.environ.get("GEMINI_API_KEY", "")
    
    if api_key:
        st.success("✅ API Key aktif")
    else:
        api_key = st.text_input("Gemini API Key", type="password")
        if not api_key:
            st.warning("⚠️ API Key gerekli")
    
    st.divider()
    
    # Bilgi
    with st.expander("📊 Proje Detayları"):
        st.markdown("""
        - **Dataset:** 42,804 makale
        - **Chunk:** 244,150 parça
        - **Model:** Gemini 2.5 Flash
        - **Vector DB:** FAISS
        """)
    
    # Temizle
    if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# API kontrolü
if not api_key:
    st.info("👈 Lütfen API Key giriniz")
    st.stop()

# Modelleri yükle
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    with open('faiss_index.pkl', 'rb') as f:
        data = pickle.load(f)
    
    return embedding_model, data['index'], data['chunks'], data['metadatas']

try:
    with st.spinner("🔄 Sistem başlatılıyor..."):
        embedding_model, index, chunks, metadatas = load_models()
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
except Exception as e:
    st.error(f"❌ Hata: {e}")
    st.stop()

# RAG fonksiyonu - CHATBOT TARZI
def get_response(question):
    # Embedding
    q_emb = embedding_model.encode([question])[0]
    q_emb = np.array([q_emb]).astype('float32')
    
    # FAISS arama
    distances, indices = index.search(q_emb, 5)
    
    # Context
    context_parts = []
    sources = []
    
    for idx in indices[0]:
        context_parts.append(chunks[idx][:500])
        sources.append(metadatas[idx]['title'])
    
    context = "\n\n".join(context_parts)
    
    # CHATBOT PROMPT
    # CHATBOT PROMPT
    prompt = f"""Sen uzman bir tıp doktorusun. Aşağıdaki bilgileri AKILLICA kullan:

KAYNAK BİLGİLER:
{context}

KULLANICI SORUSU: {question}

**AKILLI KAYNAK KULLANIMI:**
- Kaynaklar GERÇEKTEN ilgiliyse onlara dayan
- Kaynaklar ilgisizse KENDİ TIBBİ BİLGİNİ kullan
- Asla "kaynaklarda bilgi yok" deme
- Her zaman yardımcı olmaya çalış

**KURALLAR:**
1. Önce empati kur ("Geçmiş olsun")
2. Olası nedenleri sırala
3. Tedavi yöntemlerini açıkla
4. Ne zaman doktora gitmeli belirt
5. Pratik öneriler ver

DETAYLI CEVAP:"""
    # Gemini cevap
    response = model.generate_content(prompt)
    
    return response.text, list(set(sources))[:3]

# Chat arayüzü
st.markdown("### 💬 Sohbet")

# Geçmiş mesajlar
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("📚 Kaynaklar"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.caption(f"{i}. {src}")

# Yeni mesaj
if prompt := st.chat_input("Sağlık hakkında soru sorun..."):
    # Kullanıcı mesajı
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Bot cevabı
    with st.chat_message("assistant"):
        with st.spinner("💭 Düşünüyorum..."):
            try:
                answer, sources = get_response(prompt)
                st.markdown(answer)
                
                with st.expander("📚 Kaynaklar"):
                    for i, src in enumerate(sources, 1):
                        st.caption(f"{i}. {src}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"❌ Hata: {e}")

# İlk açılışta örnekler
if len(st.session_state.messages) == 0:
    st.divider()
    st.markdown("**💡 Örnek Sorular:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.button("🩺 Diyabet nedir?", use_container_width=True)
    with col2:
        st.button("💊 Migren nasıl geçer?", use_container_width=True)
    with col3:
        st.button("🤰 Hamilelik testleri", use_container_width=True)