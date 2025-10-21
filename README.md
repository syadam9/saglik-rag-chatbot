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

## 🌐 Web Arayüzü & Kullanım Kılavuzu

### 🚀 Uygulama:
**👉 https://huggingface.co/spaces/SYAdaM9/turkish-medical-chatbot**

![ ](https://github.com/syadam9/saglik-rag-chatbot/blob/main/assets/gif1.gif?raw=true)


*GIF: Chatbot'un kullanımı gösterilmektedir*

### Özellikler:

- ✅ **Chat Geçmişi:** Önceki sorularınızı görebilirsiniz
- ✅ **Kaynak Referansları:** Her cevap hangi makaleden geldiğini gösterir
- ✅ **Temizle Butonu:** Sohbeti sıfırlayabilirsiniz
- ✅ **Örnek Sorular:** Hızlı başlamak için hazır sorular

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
| Cevap Süresi | 8-10 saniye |
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
"""Sen uzman bir tıp doktorusun. Aşağıdaki bilgileri AKILLICA kullan:
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
```
## ⚠️ Sistem Sınırlılıkları

### Veri Seti Dağılımı:
- **Güçlü Alanlar:** Psikoloji (%18), Kadın Hastalıkları (%11), Beslenme (%10)
- **Zayıf Alanlar:** Dahiliye (%1.8), Kardiyoloji, Nöroloji

### Performans Notları:
- Nadir uzmanlık alanlarında tutarsızlık olabilir
- Kaynak gösterme mekanizması geliştirme aşamasında
- Sistem akıllı fallback ile her soruya cevap vermeye çalışır
---

## 🤝 Katkıda Bulunma

Bu proje Akbank GenAI Bootcamp kapsamında geliştirilmiştir.

---

---

**Seymen Sezgin**
**Uygulama Linki** : https://huggingface.co/spaces/SYAdaM9/turkish-medical-chatbot
**Kaggle Notebook Linki** : https://www.kaggle.com/code/seymensezgin/turkish-medical-rag-chatbot

---

---

