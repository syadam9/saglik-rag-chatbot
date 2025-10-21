# 🏥 Türkçe Sağlık Bilgi Asistanı - RAG Chatbot

**Akbank GenAI Bootcamp Projesi**

RAG (Retrieval Augmented Generation) mimarisi ile geliştirilmiş, 43.000+ Türkçe tıbbi makaleden oluşan veri seti üzerinde eğitilmiş bir sağlık asistanıdır. Kullanıcıların tıbbi sorularına güvenilir ve kaynaklı cevaplar sunarak sağlık okuryazarlığını artırmayı hedefler.

Uygulama, karmaşık tıbbi bilgilere erişim zorluğunu çözerek, kullanıcıların sağlık sorularına anında ve güvenilir yanıtlar alabilmelerini sağlar.
---

## 🎯 Proje Amacı

Bu proje, kullanıcıların sağlık konularında güvenilir bilgiye hızlı ve kolay erişmesini sağlamak amacıyla geliştirilmiştir. RAG mimarisi kullanılarak, doktorsitesi.com'dan toplanan 42,804 Türkçe tıbbi makale üzerinden:

- ✅ Kullanıcı sorularına doğru ve kaynağa dayalı cevaplar verilmesi
- ✅ Tıbbi bilgiye demokratik erişimin sağlanması
- ✅ Yapay zeka destekli sağlık bilgi sisteminin geliştirilmesi

hedeflenmiştir.

---

## 📊 Veri Seti Hakkında

**Dataset:** `umutertugrul/turkish-medical-articles` (Hugging Face)

### Veri Kaynağı:
- **Platform:** doktorsitesi.com
- **İçerik:** Lisanslı sağlık profesyonelleri tarafından yazılmış Türkçe tıbbi makaleler
- **Toplam Makale:** 42,804 adet
- **Format:** Parquet (.parquet)
- **Boyut:** ~110 MB

### Veri Yapısı:
```
- url: Makalenin web adresi
- title: Makale başlığı
- text: Makale içeriği (tam metin)
- name: Yazar adı
- branch: Tıp dalı (Kadın Hastalıkları, Göz Hastalıkları vb.)
- publish_date: Yayın tarihi
- scrape_date: Veri toplama tarihi
```

### Veri İşleme:
1. **Temizleme:** Boş text değerleri ve 100 karakterden kısa makaleler filtrelendi
2. **Final Dataset:** 41,788 temiz makale
3. **Chunking:** Her makale 1000 karakterlik parçalara bölündü (200 karakter overlap)
4. **Toplam Chunk:** 244,150 adet

---

## 🛠️ Kullanılan Yöntemler

### 1️⃣ **RAG (Retrieval Augmented Generation) Mimarisi**

RAG, büyük dil modellerinin (LLM) doğruluğunu artırmak için kullanılan bir tekniktir:
```
Kullanıcı Sorusu → Embedding → Vector Arama → İlgili Dökümanlar → LLM → Cevap
```

**Avantajları:**
- ✅ Güncel bilgiye erişim (model eğitimi gerektirmez)
- ✅ Kaynak referansları (hangi makaleden geldiği belli)
- ✅ Halüsinasyon azaltma (LLM uydurmuyor, kaynaktan okuyor)

---

### 2️⃣ **Teknoloji Stack**

#### **Embedding Model:**
- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Dil Desteği:** Türkçe dahil 50+ dil
- **Embedding Boyutu:** 384 boyutlu vektörler
- **Kullanım:** Metinleri sayısal vektörlere çevirme

#### **Vector Database:**
- **Teknoloji:** FAISS (Facebook AI Similarity Search)
- **Index Tipi:** IndexFlatL2 (L2/Euclidean mesafe)
- **Vektör Sayısı:** 244,150 chunk embedding'i
- **Arama Hızı:** Milisaniyeler içinde benzer metinleri bulma

#### **Language Model:**
- **Model:** Google Gemini 2.5 Flash
- **Kullanım:** Context'e dayalı cevap üretme
- **API:** Google Generative AI

#### **Web Framework:**
- **Framework:** Streamlit 1.31.0
- **Deployment:** Hugging Face Spaces
- **Arayüz:** Chat tarzı, mesaj geçmişi destekli

---

### 3️⃣ **RAG Pipeline Detayları**
```python
# 1. SORU ALMA
user_question = "Migren ağrısı nasıl geçer?"

# 2. EMBEDDING (Soruyu vektöre çevirme)
question_vector = embedding_model.encode(user_question)  # 384 boyutlu vektör

# 3. BENZER CHUNKLARI BULMA (FAISS)
similar_chunks = faiss_index.search(question_vector, top_k=5)
# En benzer 5 makale parçası bulunur

# 4. CONTEXT OLUŞTURMA
context = "\n".join(similar_chunks)

# 5. LLM'E PROMPT GÖNDERME
prompt = f"""
Kaynaklardan bilgiler: {context}
Kullanıcı sorusu: {user_question}
Kısa ve öz cevap ver.
"""

# 6. CEVAP ÜRETME
answer = gemini.generate(prompt)
```

---

### 4️⃣ **Performans Metrikleri**

| Metrik | Değer |
|--------|-------|
| Toplam Makale | 42,804 |
| Temiz Makale | 41,788 |
| Chunk Sayısı | 244,150 |
| Embedding Süresi | ~3 dakika (GPU) |
| Arama Hızı | <100ms |
| Cevap Süresi | 2-4 saniye |
| FAISS Index Boyutu | 561 MB |

---


---

## 🎨 Elde Edilen Sonuçlar

### ✅ Başarılar:

1. **Yüksek Doğruluk:** RAG mimarisi sayesinde kaynak referanslı, doğru cevaplar
2. **Hızlı Yanıt:** FAISS indexi ile milisaniyeler içinde ilgili bilgi bulma
3. **Türkçe Desteği:** 42K+ Türkçe makale ile zengin içerik
4. **Kullanıcı Dostu:** Chat arayüzü ile kolay kullanım
5. **Ölçeklenebilir:** Yeni makaleler kolayca eklenebilir

### 📊 Örnek Sorgular:

**Soru 1:** "Hamilelerde yapılan testler nelerdir?"

**Soru 2:** "Migren ağrısı nasıl geçer?"

---

## 🌐 Web Arayüzü & Kullanım Kılavuzu

### 🚀 Canlı Demo:
**👉 [Turkish Medical Chatbot - Hugging Face Spaces](https://huggingface.co/spaces/KULLANICI_ADIN/turkish-medical-chatbot)**

### Kullanım Adımları:

1. **Linke Tıkla:** Yukarıdaki Hugging Face Spaces linkine git
2. **Bekle:** İlk açılışta model yüklenir (~10 saniye)
3. **Soru Sor:** Alt taraftaki chat kutusuna sağlık sorunuzu yazın
4. **Cevap Al:** 2-4 saniye içinde kaynaklı cevap gelir
5. **Kaynakları Gör:** "📚 Kaynaklar" bölümünü açarak hangi makalelerden bilgi geldiğini görün

### Özellikler:

- ✅ **Chat Geçmişi:** Önceki sorularınızı görebilirsiniz
- ✅ **Kaynak Referansları:** Her cevap hangi makaleden geldiğini gösterir
- ✅ **Temizle Butonu:** Sohbeti sıfırlayabilirsiniz
- ✅ **Örnek Sorular:** Hızlı başlamak için hazır sorular

### 🎥 Demo GIF:

![Chatbot Demo](demo.gif)

*GIF: Chatbot'un kullanımı gösterilmektedir*

---

## 📂 Proje Yapısı
```
turkish-medical-rag-chatbot/
│
├── turkish_medical_rag_chatbot.ipynb  # Ana notebook (tüm adımlar)
├── app.py                              # Streamlit web uygulaması
├── requirements.txt                    # Python kütüphaneleri
├── faiss_index.pkl                     # FAISS vector database (Hugging Face'te)
├── demo.gif                            # Kullanım demo'su
└── README.md                           # Bu dosya
```

---

## 🔧 Teknik Detaylar

### Chunking Stratejisi:
```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Her chunk max 1000 karakter
    chunk_overlap=200,      # 200 karakter örtüşme
    separators=["\n\n", "\n", " ", ""]  # Önce paragraf, sonra cümle
)
```

### FAISS Index Oluşturma:
```python
dimension = 384  # Embedding boyutu
index = faiss.IndexFlatL2(dimension)  # L2 mesafe
index.add(embeddings_array)  # 244,150 vektör eklendi
```

### Prompt Engineering:
```python
"""Sen samimi, yardımsever bir Türkçe sağlık asistanısın.

KAYNAKLARDAN BİLGİLER: {context}
KULLANICI SORUSU: {question}

KURALLAR:
- Kısa ve öz cevap ver (5-6 cümle)
- Madde işaretleri kullan
- Kaynaklarda bilgi yoksa belirt
"""
```

---

## 🤝 Katkıda Bulunma

Bu proje Akbank GenAI Bootcamp kapsamında geliştirilmiştir.

---

## 📜 Lisans

Bu proje eğitim amaçlıdır. Dataset CC BY 4.0 lisansı altındadır.

---

## 👨‍💻 Geliştirici

**[ADIN SOYADIN]**
- GitHub: [@KULLANICI_ADIN](https://github.com/KULLANICI_ADIN)
- LinkedIn: [linkedin.com/in/PROFILIN](https://linkedin.com/in/PROFILIN)

---

---

**⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!**
