# Çoklu ortam arama benchmark'ı: bütün geçerli varyantlar

Tarih: 7 Ekim 2026. Donanım: M4 Max, 128 GB unified memory.
Durum: **Deney planı. Bu yeni deneyler çalıştırılmadı; bu dosyada sonuç veya tahmini başarı yüzdesi yok.**

Amaç: EmbeddingGemma 2'nin metin, görüntü, ses ve videodaki katkısını; metne dönüştürülmüş içerik üzerinden BM25/BGE araması, uzman modeller, hibrit arama ve yeniden sıralamayla karşılaştırmak. Kod metin görevidir. İlk aşamada etiketli kaynak bulmayı, uygun veri kümelerinde ayrıca cevap üretmeyi ölçeriz.

## 1. Kapsam ve “bütün varyantlar” tanımı

Her görevde gerçekten uygulanabilen arama kanallarının **bütün boş olmayan alt kümeleri** çalıştırılır. Kanalların sırası yeni deney oluşturmaz: B+G ve G+B aynı yöntemdir. Her alt küme dört sıralama koşuluyla değerlendirilir. Parametre duyarlılığı ve veri temsili deneyleri ayrıca tanımlanmıştır; bu eksenler sessizce ana sonuçlara karıştırılmaz.

| Kod | Arama kanalı | Girdi |
|---|---|---|
| B | BM25 | Kaynaktan çıkarılan veya kaynak için üretilen sabit metin |
| G | BGE-M3 dense | B ile aynı metin |
| E | EmbeddingGemma 2 metin | B ile aynı metin |
| N | EmbeddingGemma 2 doğrudan medya | Görüntü, dalga biçimi, video kareleri; sorgu görevine göre metin veya bileşik girdi |
| S | Alan uzmanı | Fotoğrafta SigLIP2; PDF'de ColQwen2.5; çevresel seste CLAP; videoda CLIP kare havuzlama |
| J | EmbeddingGemma 2 birleşik girdi | Aynı kaynağın medyası + ondan türetilmiş metin, tek model girdisinde |

J bir RRF birleşimi değildir: içerikler modele birlikte girer, tek temsil üretilir. N+E ise iki ayrı aramanın sıralamalarını birleştirir. Bu iki yolu ayrıca karşılaştıracağız.

| Görev | Kanallar | Arama kombinasyonu | Dört sıralama koşuluyla yöntem ailesi |
|---|---|---:|---:|
| Fotoğraf | B,G,E,N,S,J | 63 | 252 |
| Görsel belge | B,G,E,N,S,J | 63 | 252 |
| Çevresel ses | B,G,E,N,S,J | 63 | 252 |
| Konuşma | B,G,E,N,J | 31 | 124 |
| Video, görsel ana koşul | B,G,E,N,S,J | 63 | 252 |
| Kod | B,G,E | 7 | 28 |
| Görüntü + değişiklik talimatı | B,G,E,N,J | 31 | 124 |
| **Toplam** | | **321** | **1.284** |

Bunlar yöntem aileleridir; aynı aile farklı veri koleksiyonları, diller ve aday bütçelerinde yeniden değerlendirilir. CSV 1.284 aileyi tek tek listeler, sonuç içermez. Desteklenmeyen bir çalışma başarılı gibi sayılmaz; nedenli `unsupported` veya `failed` durumu alır.

Dört sıralama koşulu: (1) yeniden sıralama yok; (2) Laya'nın metin üzerinden ilgili/ilgisiz olasılığı; (3) ayrı bir model olan BGE-reranker-v2-m3; (4) yerel Gemma 4 ile sabit, noktasal kaynak-ilgililiği puanı. Dördüncü koşul bir hakem ölçütü değildir: sıralanan sonuçlar yine veri setinin etiketleriyle puanlanır.

## 2. Her testte girdi, çıktı ve karşılaştırma

### Fotoğraf: XM3600

- Arşiv: 3.600 görüntü; Türkçe 7.233, İngilizce 7.200 açıklama sorgusu. Diller ayrı raporlanır.
- Girdi örneği: “Kırmızı şemsiyeyle yürüyen kişi.” Çıktı: sıralı fotoğraf kimlikleri.
- B/G/E metni: her aday fotoğraf için sorguları görmeden, sabit yerel Gemma 4 ile üretilmiş betimleme. Alt deneyde OCR metni eklenir.
- N: gerçek görüntü. S: SigLIP2 görüntü/metin eşleşmesi. J: gerçek görüntü + üretilmiş betimleme.
- Girdi deneyleri: görüntü; OCR; üretilmiş betimleme; bunların yedi boş olmayan birleşimi. OCR'si boş olanlar ayrıca sayılır; yapay dolgu metni üretilmez.
- Yönler: metin→görüntü ana görev; görüntü→metin ayrı görev. Ters yönde aynı görüntüye ait bütün açıklamalar ilgili sayılır. Sadece caption sorgusunu ters çevirmek yeni bağımsız niyet üretmez.
- Ana rapor: R@1/5/10 (tek etiketli eşleşmede isabet), MRR@10, TR–EN farkı, nesne/eylem/sayı/ilişki hata örnekleri.

Kaynak: [XM3600 proje](https://google.github.io/crossmodal-3600/), [makale ve dil istatistikleri](https://arxiv.org/pdf/2205.12522).

### Görsel belge: ViDoRe V3'ün sekiz açık koleksiyonu

- Makalede 19.256 sayfa bildiriliyor; sabitlenen indirilebilir sürümde enerji koleksiyonunun dört eksik kaydı nedeniyle **19.252 sayfa** ve çeviriler hariç **2.419 temel sorgu** doğrulandı. Sekiz arşiv ayrı aranır. Genel makro ortalama ayrıca sunulur.
- Girdi: sorulan bilgiyi içeren sayfayı bulma sorgusu. Çıktı: PDF/sayfa kimlikleri.
- B/G/E: veri kümesinin yayımladığı çıkarılmış markdown. N: sayfa görüntüsü. S: ColQwen2.5. J: görüntü + aynı markdown.
- Ek temsil koşulları: yayımlanmış markdown; kendi sabit yerel OCR çıktımız; kaynaktan çıkarılabilen doğal PDF metni. Bunlar birbirinin yerine sessizce geçirilmez.
- Sayfa görüntüsü / çıkarılan metin / sorgudan bağımsız üretilmiş sayfa özeti için yedi girdi birleşimi. Çok uzun sayfalarda önceden tanımlanmış segmentler kullanılır; segment skoru sayfa kimliğine sabit kuralla birleştirilir.
- Metin ağırlıklı, tablo, grafik ve şema örnekleri ayrı analiz edilir. Hazır olmayan kategori etiketleri oluşturulursa yöntemi açıklanır.
- Ana metrik nDCG@10; ayrıca Hit@5, gerçek çok-pozitif Recall@5/10/20. Kısmen ilgili ve tamamen ilgili etiketleri korunur.
- Türkçe çeviri uzantısı oluşturulursa insan anlam kontrolü ve ayrı isim kullanılır; resmî Türkçe ViDoRe sonucu diye sunulmaz.

Kaynak: [ViDoRe V3 veri ve protokol](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3).

### Çevresel ses: Clotho v2.1 değerlendirme bölümü

- 1.045 kayıt, 5.225 açıklama sorgusu. Sesler 15–30 saniyedir.
- Girdi: “Yağmur ve gök gürültüsü duyulan kayıt.” Çıktı: ses dosyası kimlikleri.
- B/G/E: ham kayıttan üretilmiş ses olayı betimlemesi. N: ham ses. S: CLAP. J: ham ses + üretilmiş betimleme.
- Ek temsil koşulları: ham ses / üretilmiş ses betimlemesi / gerçek ASR çıktısı; yedi birleşim. ASR konuşmasız kayıtta boş kalabilir; bu koşul ses olayını yazıya dökme modeliyle aynı sayılmaz.
- Özgün eşleşmelerde R@1/5/10. Aynı kayıtlar için sağlanan 1.069 çok-ilgili sorguda ayrıca resmî mAP@16.
- Ses betimlemelerini üretme süresi toplam indeks hazırlama maliyetine eklenir.

Kaynak: [DCASE 2025 Clotho protokolü](https://dcase.community/challenge2025/task-language-based-audio-retrieval), [sabit medya arşivi](https://zenodo.org/records/4783391).

### Türkçe konuşma: FLEURS tr_tr

- Testte 743 kayıt vardır. Bu sayı 743 benzersiz semantik sorgu demek değildir.
- Aynı cümle/normalize edilmiş transkripte ait kayıtlar pozitif grup yapılır. Ortaya çıkan benzersiz sorgu sayısı manifestte gerçek sayımla kaydedilir.
- B/G/E: Whisper'ın kayıttan çıkardığı transkript. N: ham konuşma sesi. J: ham ses + ASR transkripti. Yapay bir uzman kanal eklenmez.
- İki sorgu türü: gerçek transkriptle birebir içerik arama; insan kontrolünden geçmiş yeniden ifade edilmiş sorguyla anlam arama. İkincisi türetilmiş uzantıdır.
- Ham ses / ASR transkripti / ASR'den üretilmiş kısa anlam özeti için yedi birleşim ayrı temsil deneyi olur.
- Sesli sorgu→metin yönü de etiketli cümle gruplarıyla ölçülür; sorgu tarafında doğrudan ses embedding'i ile Whisper→metin yolu karşılaştırılır.
- Retrieval ölçütlerine ek WER, yani yazıya çevirme hatası, raporlanır. Gold transkript birebir sorgu, pozitif grup tanımı ve WER hesabında kullanılabilir; aday metnine ASR sonucu diye konmaz.

Kaynak: [Google FLEURS](https://huggingface.co/datasets/google/fleurs), [Whisper](https://github.com/openai/whisper).

### Video: MSR-VTT 1K-A

- 1.000 klip ve protokolün belirlediği 1.000 sorgu. Tam MSR-VTT ile karıştırılmaz.
- Görsel ana koşulun B/G/E metni: aynı örneklenmiş karelerden üretilen video betimlemesi ve ekranda okunan metin. Ses bu ana koşula karışmaz.
- N: EmbeddingGemma ile kare dizisi; S: CLIP kare temsillerini sabit yöntemle havuzlama; J: kare dizisi + üretilmiş metin.
- Ses kanalı bulunan kliplerde ayrı eşleştirilmiş AV uzantısı: kareler, ham ses, ASR transkripti, görsel betimleme. Dört girdinin 15 boş olmayan alt kümesi. Aynı AV klip kümesinde görsel-only referans da tekrar ölçülür.
- Ses dosyası olmayan kliplerin yerine altyazı veya gold açıklama geçirilmez. Kullanılabilir medya manifesti açıklanır.
- FPS 0,5/1/2; azami kare 8/16/32 koşulları ayrı parametre deneyidir. Kare kırpma, eşit aralıklı örnekleme ve bağlam bütçesi kaydedilir.
- Ana sonuç klip bulmadır. Zaman damgası etiketli ayrı veri olmadan “doğru saniyeyi bulma başarısı” iddia edilmez.
- Ters eylem sırası ve kısa olaylar için ek zorluk seti hazırlanırsa ayrı etiketlenir.

Kaynak: [MSR-VTT protokol dosyaları](https://github.com/ArrowLuo/CLIP4Clip), [Google medya işleme kılavuzu](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers).

### Kod: temizlenmiş CodeSearchNet

| Dil | Test sorgusu | Aday kod |
|---|---:|---:|
| Python | 14.918 | 43.827 |
| JavaScript | 3.291 | 13.981 |
| Java | 10.955 | 40.347 |
| Go | 8.122 | 28.120 |
| PHP | 14.014 | 52.660 |
| Ruby | 1.261 | 4.360 |

- Altı dilde toplam 52.561 sorgu, ayrı galerilerde toplam 183.295 aday. Bu toplam ortak arşiv boyutu değildir.
- Girdi: fonksiyonun işiyle ilgili doğal dil sorgusu. Çıktı: fonksiyon kimliği.
- Yedi arama: B, G, E, B+G, B+E, G+E, B+G+E. Her biri dört sıralama koşuluna girer.
- Bütün modeller aynı temizlenmiş kodu görür; sorgunun kaynak docstring'i adaydan çıkarılmış olmalıdır. Kodun API/identifier isimleri korunur.
- BM25 için orijinal identifier ve snake_case/camelCase parçaları sabit tokenizasyonla kullanılır.
- Ek deneyler: genel arama öneki / EG2 kod arama öneki; kaynak kod / kaynak koddan üretilmiş açıklama / ikisi birlikte. Üretilmiş açıklama gold docstring değildir.
- İngilizce resmî sorgular ana sonuçtur. Türkçe çeviriler insan kontrolünden sonra ayrı uzantı olur.
- MRR@10, nDCG@10 ve Hit@1/5/10; dil başına skor ve makro ortalama.

Kaynak: [Microsoft temizlenmiş veri ve istatistikleri](https://github.com/microsoft/CodeBERT/blob/master/GraphCodeBERT/codesearch/README.md).

### Birleşik sorgu: CIRR

- Girdi: referans görüntü + “aynısı ama köpek koşuyor” gibi değişiklik talimatı. Çıktı: hedef görüntü.
- Tamamen yerel, etiketleri açık validation bölümü kullanılır ve açıkça validation skoru diye adlandırılır. Kapalı test sunucusuna ait skor iddia edilmez. Ayarlar train içindeki ayrı geliştirme verisinde belirlenir.
- B/G/E sorgusu: referans görüntüden sorgudan bağımsız üretilmiş betimleme + değişiklik talimatı. Aday tarafı üretilmiş betimlemedir.
- N sorgusu: gerçek referans görüntü + talimat birlikte; aday gerçek görüntüdür.
- J: aynı tam sorgu; aday görüntü + adayın üretilmiş betimlemesi.
- Sadece referans görüntü / sadece değişiklik yazısı / ikisi birlikte sorgu ablasyonları ayrı gösterilir. Bunlar eşdeğer bilgiye sahip yöntemler değildir.
- Referans görüntü sonuç adaylarından çıkarılır. Tam galeri Recall ve resmî küçük grup Recall birbirine karıştırılmaz.

Kaynak: [CIRR resmî protokol](https://github.com/Cuberick-Orion/CIRR).

## 3. Yeniden sıralama ve Gemma'nın rolleri

- İlk aşama arşivden aday bulur. Yeniden sıralayıcı yalnız o adayların yerini değiştirebilir.
- Her arama varyantının aynı aday kimlikleri dört sıralama koşuluna verilir. Ana aday sayısı 50; ek koşullar 20 ve 100. İlk 1/5/10/20 sonuç ölçülür.
- Laya ve BGE-reranker metin görür. Fotoğraf/ses/video için kaynakta üretilmiş aynı metin kullanılır; “ham medyayı değerlendirdi” denmez.
- Gemma 4 kaynak ilgililiği koşulunda kaynağın gerçek desteklenen girdisini görür. Görsel testte görüntü; seste dalga biçimi; kodda kod. Sorgu bağlamı kaybolmaz: CIRR'de referans + talimat birlikte verilir.
- Görevler arası tutarlılık için medya kabul eden aynı yerel Gemma 4 E4B checkpoint'i tercih edilir. Model revision'ı ve yerel backend desteği küçük ön testte doğrulanıp sabitlenir. Metin/görüntü destekli mevcut 26B A4B'ye ham ses desteği atfedilmez.
- Gemma'nın betimleme üretmesi, kaynakları sıralaması ve son cevap yazması üç ayrı rol ve üç ayrı maliyet olarak kaydedilir.
- Aday metni sığmıyorsa sabit segmentleme kullanılır; sessiz token kırpması yapılmaz. Laya bir uzman reranker diye varsayılmaz; sabit ilgililik sorusuyla deneysel karar modeli olarak kullanılır.
- Jev istenen çevrimiçi karşılaştırmada aynı sabit aday metinlerini görebilir; yerel çekirdek matrisine dahil değildir. Ayrı model/istek/maliyet kaydı gerekir; bu plan API çağrısı başlatmaz.

Kaynaklar: [BGE reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3), [Laya](https://huggingface.co/convaiinnovations/laya-multilingual), [Gemma 4 yetenekleri](https://ai.google.dev/gemma/docs/core/model_card_4).

## 4. Ortak ayarlar ve kontrollü ek deneyler

Ana koşul: EG2 768 boyut, normalize edilmiş vektörler, yaklaşık arama kullanmadan kesin benzerlik araması; çok kanallı sıralamada eşit ağırlıklı RRF, k=60. Her kanal en iyi 100 adayını sağlar; ID bazlı tekilleştirme sonrası son sıralamadan ilk 20/50/100 seçilir. Kararlı eşit-skor kuralı ID'dir.

Daha çok kanal daha çok hesaplama ve daha geniş ham aday birleşimi yaratır. Bunun yanında **toplam ilk-aşama aday bütçesi 100'e eşitlenmiş** kontrol çalıştırılır; bütçe kanallara önceden sabit paylaştırılır. Kalite ve hesaplama maliyeti beraber raporlanır.

Eşit bütçe kontrolünde 100, tekilleştirme öncesi alınan toplam kanal sonucudur. Tekilleştirme sonrası 100'den az aday kalırsa sessizce yeni aday çekilmez; gerçek benzersiz aday sayısı yazılır ve yeniden sıralayıcı min(K, benzersiz aday sayısı) öğe görür. Ek adayla tamamlama yapılacaksa ayrı, ek hesaplaması ölçülen koşul olur.

| Eksen | Değerler | Uygulama |
|---|---|---|
| EG2 vektör boyutu | 128,256,512,768 | Kırpma sonrası yeniden normalize et; EG2 içeren sıralamaları tekrar türet |
| Görsel token bütçesi | 70,140,280,560,1120 | Görsel içeren N/J kanallarında; toplam bağlam bütçesini aşanlar açıkça uygulanamaz |
| Video örnekleme | 0,5/1/2 FPS; max 8/16/32 kare | Geçerli kombinasyonlar; ses koşulu aynı sabit tutulur |
| Rerank aday sayısı | 20,50,100 | Bütün geçerli yöntem ailelerinde |
| BGE-M3 iç işlevleri | dense; sparse; multi-vector; üçünün 7 alt kümesi | Ayrı BGE karşılaştırması; ana G=dense sonucunu sessizce değiştirme |
| Sorgu dili | Hazır TR/EN ve diğer resmî diller | Çeviri uzantılarını ayrı göster |
| Arşiv ölçeği | Resmî tam galeri; uygun görevde 10K/100K genişletme | Etiketli pozitifleri koru, sabit negatif manifesti; resmî skorlarla karıştırma |
| Sayısal hassasiyet | Desteklenen BF16; FP32 referans | EG2 için FP16 yok; cihaz ve backend kaydı zorunlu |

Yöntem altkümeleri tam kapsamdır. Ayar etkileri ayrı, diğer eksenler sabit tutularak ölçülür; bütün ayarların kontrolsüz devasa çapraz çarpımı ana tablo diye sunulmaz. Gereken etkileşim deneyleri (video FPS×kare sınırı gibi) önceden belirtilir. Tek parametre sonucu bütün kombinasyonların sonucu yerine geçmez.

Kaynaklar: [EG2 model kartı](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2), [BGE-M3 işlevleri](https://huggingface.co/BAAI/bge-m3).

## 5. Son cevap kalitesi

Kaynak bulma ile cevap yazma ayrı raporlardır. Fotoğraf/ses/video/kod arama veri kümelerine yapay QA cevap doğruluğu yüzdesi eklenmez.

Gold cevabı olan PDF sorularında, bütün retrieval/rerank varyantlarının ilk 5 kaynağı aynı sabit Gemma üreticisine verilir. Aynı soru, prompt, üretim ayarları ve kanıt bütçesi kullanılır. Sayfa metni ve gerçek sayfa görüntüsüyle cevap üretme ayrı temsil koşullarıdır.

Üç referans: kaynaksız Gemma; gold kaynak verilmiş Gemma; bulunan kaynak verilmiş Gemma. Gold kaynak koşulu normal sisteme sızdırılmaz, yalnız generator kapasitesini açıklayan oracle kontrolüdür. Çıktıda cevap ve kaynak kimlikleri bulunur.

Kelime F1/EM ve kaynak atıf doğruluğu ölçülür; anlam doğruluğu insan etiketli sabit denetim setiyle ayrıca kalibre edilir. Gemma'nın kendi cevaplarını tek başına doğru ilan etmesi ölçüt değildir. Jev/güçlü LLM karşılaştırması yapılırsa insan referansının yerine varsayımsal “gerçek doğruluk” diye sunulmaz.

## 6. Veri, sızıntı, ölçüm ve paylaşım

- Her model aynı görevde aynı aday arşivini ve aynı sorguları kullanır; testte başarısız kayıtlar sessizce atılmaz.
- Gold caption, cevap, query'ye bağlı kutu işareti ve gold transkript aday metninin üretimine giremez.
- Üretilmiş metin sorgudan bağımsız, tek sabit model/prompt/seed ile hazırlanır. B/G/E aynı metni paylaşır. Uzun içerik segmentleme ve sayfa/klip birleştirme kuralları dondurulur.
- Çok-pozitif görevlerde Hit@K = en az bir pozitif bulundu mu; Recall@K = ilgili öğelerin ne kadarı bulundu. İki ölçüt ayrı sütunlardır.
- Her görevin resmî ana metriği korunur. nDCG, MRR ve isabet yüzdeleri tek “genel başarı” sayısına karıştırılmaz.
- Aynı medya/belgeden türeyen sorgular gruplanarak eşleştirilmiş bootstrap güven aralığı hesaplanır. Kazanılan/kaybedilen sorgular ve farkın büyüklüğü gösterilir.
- p50/p95 sorgu süresi, metin üretimi/embedding/index/rerank süreleri, peak bellek ve disk alanı ayrı yazılır. Sıcak ölçümde cache'den cevabı okumak inference hızı diye sayılmaz.
- Küçük 20–50 kayıtlı koşu yalnız teknik sağlık ve hız kontrolüdür. Ana sonuçlar ilgili matris hücresinin tüm dondurulmuş test sorgularıyla oluşur.
- Sorgu/aday hash'leri, veri/model revision'ları, cihaz, paket sürümleri, bütün ayarlar, çıktılar ve başarısızlıklar manifestte tutulur.
- Açık veri setleri için model eğitiminde hiç görülmedi garantisi verilmez; bilinen eğitim/ayar örtüşmeleri belirtilir. Yeni, insan etiketli zorluk seti ayrı dış doğrulama olur.
- Kod, ayarlar, skorlar, aday ID listeleri ve inceleme arayüzü yayımlanır. Ham medyanın yeniden dağıtımı kaynak lisanslarına göre yapılır.

## 7. Çalıştırma düzeni ve hesaplamayı paylaşma

1. Veri lisansı/medya erişimi, model revision'ı ve yerel modalite desteğini doğrula.
2. Sorgu/aday manifestlerini dondur; hazırlık ve test verisini ayır.
3. OCR, ASR ve betimlemeleri bir kez üret; her varyant için yeniden üretme.
4. Her model/girdi/ayar için embedding'leri ve temel sıralamaları bir kez hesapla.
5. Bütün altküme RRF sıralamalarını bu önbellekten üret.
6. Sorgu–aday–kanıt–model–prompt düzeyindeki noktasal rerank skorlarını tekrar kullan. Listeye bağlı yöntem kullanılırsa tüm liste cache anahtarına girer.
7. Tam testleri bitir; tamamlanmamış hücreleri açıkça göster. Desteklenmeyen hücreye skor uydurma.
8. Resmî veri sonuçları, Türkçe türetilmiş uzantılar, ölçek testleri ve QA sonuçlarını ayrı sekmelerde yayımla.

Tahmini süre küçük hız ölçümünden hesaplanacaktır; henüz toplam saat/gün sözü verilmemiştir. 1.284 yöntem ailesi, aynı modelleri 1.284 defa sıfırdan çalıştırmak anlamına gelmez. Bununla birlikte tam aday havuzları üzerinde yeniden sıralama ve medya betimlemesi anlamlı hesaplama gerektirir.
