# Çoklu ortam benchmark — 7 Ekim 2026 sabah raporu

Gece kuyruğu, 09:00'dan önce başlamış PHP deneyi bitince **09:08 Europe/Istanbul'da kapandı**. Supervisor ve son model süreci sonlandı. **Araştırma paketinin bütünü tamamlanmadı.** Bu rapor, tamamlanan ölçümleri kısmi ve yapılmamış işlerden ayırır.

Ana matrisin **1.060 / 18.460 koşulu (%5,74)** tamamlandı; **17.400 koşul kaldı**. Bu payda 21 koleksiyonun yöntem, iki aday bütçesi ve 20/50/100 rerank aday sayısı koşullarını içerir. Aynı sonuçların yinelenen dışa aktarımları ve beş soruluk teknik kontrol sayılmaz. Ek boyut/temsil deneyleri ve cevap kalitesi çalışmaları ana paydanın dışındadır. Bu bir kapsam oranıdır; zaman veya hesaplama maliyeti yüzdesi değildir. [Sayım ve kaynak hash'leri](../reports/multimodal/progress-summary.md).

## Tamamlanan ölçümler

| Deney | Tamamlanan kapsam | Başlıca sonuç |
|---|---|---|
| Türkçe/İngilizce fotoğraf araması | Aynı 3.600 görüntü; 7.233 TR ve 7.200 EN sorgu; EG2, SigLIP2, birleşim | EG2 Hit@5: TR %84,64, EN %83,75; eşit ağırlıklı birleşim EG2'yi artırmadı |
| Çevresel ses | Clotho özgün 5.225 sorgu ve ayrı 1.037 sorguluk ek alaka protokolü | Özgün testte EG2/CLAP/birleşim Hit@5: %11,75 / %37,42 / %26,47 |
| Türkçe konuşma | 743 kayıt, 329 farklı transkript sorgusu; Whisper ve 62 arama koşulu | Whisper WER %7,29; ASR metninde BM25/BGE/EG2 ve birleşik ses+metinde EG2, Hit@1 ve Recall@5'te %100 |
| Video | MSR-VTT 1K-A: 1.000 sorgu / 1.000 video; yalnız görüntü kareleri | EG2/CLIP/birleşim Hit@5: %75,00 / %53,80 / %67,10 |
| Kod | Python, Ruby, Go, Java, PHP; 49.270 sorgu; yedi yöntem × iki bütçe = 70 koşul | EG2, beş dilde iki bütçede de en yüksek gözlenen Hit@5 değerini verdi |
| Bilgisayar bilimi belgeleri | 215 sorgu / 1.360 sayfa; 126 arama + 378 Laya + 378 BGE = 882 koşul | Tek başına ColQwen en yüksek nDCG@10'u verdi; BGE bazı başlangıçları iyileştirdi, Laya ana 63 karşılaştırmanın tamamında düşüş gösterdi |
| BM25 başlangıçları | Altı kod dili ve sekiz belge koleksiyonunun tamamında 54.980 sorgu | Pozitif kelime eşleşmeleriyle tam başlangıç sonuçları yayımlandı |
| Vektör boyutu | Yedi medya görünümünde 104 ek koşul | 768 boyutun sonuçları doğrulandı; 128/256/512 önekleri tam arşivde tekrar arandı; yeni model çıkarımı yapılmadı |

Bu satırlar aynı kaynakların ve sorguların farklı görünümlerini içerebilir; sorgu sayılarını toplayıp bağımsız toplam örnek sayısı olarak kullanmayız.

## Kod araması: doğru fonksiyonu ilk 5'te bulma

| Dil | Sorgu | BM25 | BGE-M3 | EmbeddingGemma 2 |
|---|---:|---:|---:|---:|
| Python | 14.918 | %38,93 | %62,18 | %84,46 |
| Ruby | 1.261 | %45,60 | %68,20 | %86,28 |
| Go | 8.122 | %61,94 | %87,70 | %96,28 |
| Java | 10.955 | %39,42 | %63,73 | %83,80 |
| PHP | 14.014 | %33,27 | %57,74 | %75,75 |

Bunlar aynı sabit arşiv ve sorgularda, kanal başına 100 aday alınan koşuldur. Kodun referans docstring ve yorumları çıkarılmıştır. EG2 genel arama önekini kullanır; kod için özel önek ayrı denenmedi. Kullanılan eşit ağırlıklı RRF birleşimleri bu beş dilde EG2'nin Hit@5'ini artırmadı. Bu sonuç, başka birleşim stratejilerinin de başarısız olacağını göstermez. Kod üretme veya son cevap doğruluğu ölçülmedi. [Yedi yöntem, iki bütçe, MRR/nDCG, güven aralıkları ve arşiv seçim kapsamı](../reports/multimodal/code-dense-summary.md).

Son PHP koşulu 14.014 sorgu ve 52.660 fonksiyonu içerir. EG2−BGE farkı Hit@5'te **18,02 yüzde puanı**, %95 eşleştirilmiş fonksiyon-grubu aralığı **[17,26; 18,77]**. Bu aralık aynı repository içindeki fonksiyonların bağımlılığını ayrıca modellemez. Tam 14 koşuldaki 196.196 sıralama ve 3.335.332 metrik değeri, gerçek kanal sıralamalarından ve etiketlerden yeniden doğrulandı.

## Kısmi kalan bölümler

- Fotoğraf, çevresel ses ve videoda doğrudan medya/uzman karşılaştırmaları tamam; kaynağın kendisinden açıklama üretme, bu metinlerle arama ve bütün reranker birleşimleri tamamlanmadı.
- Türkçe konuşmada ASR ve arama tamam; tam Laya/BGE/Gemma reranker deneyleri tamamlanmadı.
- Kodda beş dilin araması tamam; JavaScript'te yalnız BM25 başlangıcı tamam. Kod için tam Laya/BGE/Gemma sıralama deneyleri bekliyor.
- Belgelerde sekiz koleksiyonun BM25 başlangıcı tamam; bütün kanal birleşimleri ve Laya/BGE yalnız bilgisayar bilimi koleksiyonunda tamam. Kalan yedi koleksiyonun dense/medya/birleşim/reranker deneyleri bekliyor. Tam Gemma belge koşulları da bekliyor.
- 104 boyut koşulu tamam; diğer görüntü token bütçeleri, video kare/FPS koşulları, BGE sparse/multi-vector, kod öneki ve temsil deneyleri tamamlanmadı.

## Henüz ölçülmeyen veya engelli işler

- Çoklu ortam üzerinde tam Gemma cevap üretme ve anlam doğruluğu ölçümü yok. Önceki 20 soruluk metin RAG pilotu ayrı deneydir ve bu tamamlanma oranına eklenmez.
- Beş soruluk Gemma ilgililik kontrolü yalnız teknik çalışabilirliği doğrular; bütün test kümesinde kalite sonucu değildir.
- İnsan kontrolünden geçmiş konuşma paraphrase sorguları, ters arama yönleri ve diğer planlı uzantılar tamamlanmadı.
- CIRR'nin resmî görüntülerine erişim bekliyor. Yerine başka bir galeri geçirilmedi; resmî sonuç açıklanmıyor.

## Sonuçları yorumlama sınırları

Konuşma sorguları resmî transkriptin kendisidir. ASR metninde %100 kayıt bulma, serbest sorularda veya farklı ifadelerde %100 anlam başarısı demek değildir. WER'in %7,29 olması yaklaşık her 100 referans kelime için 7,29 ekleme/silme/değiştirme anlamına gelir; 1−WER anlam/cevap doğruluğu sayılmaz. [Ayrıntılı ASR denetimi](../reports/multimodal/speech-asr-summary.md).

Belge rerankerleri sayfa metnini görür. Laya'nın düşüşünde olumlu sınıf, skor yönü ve girdi eşlemesi kontrol edildi; düşüşün nedeni henüz belirlenmedi. BGE K=50, 63 başlangıcın 60'ında nDCG'yi artırdı; ancak tek başına ColQwen nDCG 0,7585 ile bütün BGE K=50 koşullarından yüksek kaldı. CS'de tek bağlı kaynak grubu, Clotho ek alaka protokolünde baskın bir grup bulunduğundan bu görünümlerde güven aralığı verilmez. [Belge karşılaştırması](../reports/multimodal/document-reranker-summary.md).

Video koşulu ses kanalını tüketmez. Fotoğrafta SigLIP2'nin sınırı nedeniyle 56 Türkçe sorgu açıkça kısaltılmıştır. G/E'nin eksik çalışma zamanı tanıları, ölçülmüş sıfır kırpılma veya sıfır çıkarım süresi olarak sunulmaz. Boyut taramasındaki depolama oranları yalnız ham vektör alanına aittir; toplam RAM/hız iddiası değildir.

## Gece neden bütün paket bitmedi?

Plan, veri/dil/ayar genişletmeleriyle 18.460 ana koşul içeriyor; tam yeniden sıralama ve medya açıklaması üretimi ciddi ek hesaplama gerektiriyor. Bir ana modelin embedding'ini üretmek bütün varyantları tamamlamıyor. Gece ayrıca yeni GPU işlerindeki macOS Metal derleyici hizmeti erişim hatası ilerlemeyi durdurdu. Yeni süreçte sağlık kontrolleri geçti ve kuyruk 07:35'te kurtarıldı; tüm sorunun kaynağı kesin olarak teşhis edilmiş sayılmıyor.

09:00 yeni iş başlatma sınırı korundu. PHP 08:37'de başlamıştı, normal biçimde 09:08'de bitti; ardından kalan işler ertelendi. Bitmiş önbellekler, dondurulmuş veriler ve yerel günlükler korundu. Sonraki devamda aynı sonuçlar yeniden çıkarım diye sayılmadan kullanılabilir. Ham veri, medya, transkript, sorgu, prompt, hata günlüğü ve anahtar GitHub'a yüklenmedi.

[İngilizce README](../README.md) · [Türkçe README](../README.tr.md) · [Tam ilerleme sayımı](../reports/multimodal/progress-summary.md) · [Bütün yayımlanmış koşular](../reports/multimodal/README.md) · [Kalan kapsamın planı](multimodal-plan.tr.md)
