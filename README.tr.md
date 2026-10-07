# rag-benchmark

**Türkçe RAG sistemlerini aynı koşullarda, yerel modellerle karşılaştırmak için bir deney düzeneği.** BM25, BGE-M3 ve EmbeddingGemma 2 belge aramasını; Laya'nın yeniden sıralamaya katkısını; aynı Gemma 4 modelinin ürettiği cevapları ölçer.

[English](README.md) · [Deney protokolü](docs/protocol.md) · [Sonuçlar ve doğrulama durumu](docs/results.md) · [Anlamsal değerlendirme](docs/semantic-evaluation.md)

**Mevcut durum:** **20 soru × on varyant pilotu tamamlandı: 200/200 çıktı**, 37.511 parçanın tamamında arama yapıldı. İki dense indeks de hazır. Ayrı 2.000 soruluk BM25 arama deneyi tamamlandı; on varyantın 2.000 soruluk geliştirme deneyi ve 12.530 soruluk son test henüz çalıştırılmadı. [CI](https://github.com/erendikmenn/rag-benchmark/actions/workflows/ci.yml) · [Ölçümler ve sınırları](docs/results.md).

**Ek deney çalıştırıcıları:** ters arama, doğrulanmış mevcut medya vektörlerini; kod sorgusu öneki karşılaştırması mevcut belge/parça vektörlerini kullanacak. Gerçek önbellek kontrolleri geçti. Üç ters arama işi [tamamlandı ve bağımsız denetlendi](reports/ablations/reverse/README.md). Altı kod sorgu öneki işi de [tamamlandı ve bağımsız denetlendi](reports/ablations/code-query-prefix/README.md). [Komutlar ve sınırlar](docs/multimodal-running.md). Ayrı BGE sparse/ColBERT hesaplama çekirdeği CPU testlerinden geçti; altı dilin dense önbelleğini doğrulayan köprü de bağımsız CPU kontrollerini geçti. Gerçek token backend’i ve sparse/ColBERT tam galeri ölçümleri henüz tamamlanmadı. [Önbellek kanıtı](reports/multimodal-bge-baseline-reuse-readiness.json).

Belge QA için aynı tam üretim isteğini farklı arama koşullarında paylaşan isteğe bağlı önbellek de hazır. Kaynak byte’ları ve model ayarları aynı olmalı; her görevin cevabı kendi referansıyla ayrı değerlendirilir. Bu altyapı CPU testlerinden geçti; yeni QA cevap kalitesi ve gerçek çağrı tasarrufu henüz ölçülmedi. Mevcut GPU kuyruğuna altı tek soruluk QA kontrolü eklendi: kaynaksız, doğru kaynaklı ve arama sonuçlu koşulların metin ve görüntü sürümleri. İlk gerçek model kontrolünde altı tek soruluk koşulun dördü geçti; ikisinde model, verilen kaynaklar arasında bulunmayan atıf kimlikleri yazdığı için çıktı reddedildi. Yeni `qa-exact-source-id-citations-v2` istemi izinli kaynak ID’lerini açıkça listeliyor; CPU kontrolleri geçti, yeni gerçek deneme güvenli kuyruk geçişini bekliyor; bu tamamlanmış doğruluk ölçümü sayılmıyor.

## Çoklu ortam benchmark'ı — tam veri koşuları başladı

Yeni paket **fotoğraf, görsel belge, çevresel ses, Türkçe konuşma, video, kod ve görüntü+talimat sorgularını** ölçüyor. Kayıtlı kapsam **321 arama kombinasyonu × dört sıralama koşulu = 1.284 yöntem ailesi**; veri/dil bölümleri ve ek ayarlar bunun üzerine geliyor. Bunlar planlanan yöntemlerdir; **tamamlanmış deney sayısı değildir**.

**Kullanıcının devam talimatıyla deneyler 7 Ekim 13:01’de yeniden başladı.** Önceki gece saat sınırı kaldırıldı. İki GPT-6.1 Sol ajanı CPU geliştirme, veri kontrolü ve ölçüm denetimini paralel yürütüyor; ağır yerel GPU işlerini tek kuyruk yönetiyor. JavaScript embedding araması tamamlandı; Türkçe konuşmanın Laya/BGE yeniden sıralaması tamamlandı; Clotho kaynak açıklamaları ve bu galerinin 126 arama koşulu tamamlandı; ayrı ek alaka protokolü de tamamlandı. Fotoğraf/video açıklamaları çıktı token sınırına takıldı ve ayrı kimlikle kurtarma bekliyor; Fransızca enerji belgesi araması tamamlandı; Türkçe fotoğraf açıklamalarının 512 token kurtarması çalışıyor. Aşağıdaki ilerleme oranı yalnız tamamlanmış sonuçları sayar. [Güncel çalışma durumu](docs/multimodal-status.md) · [Önceki sabah özeti](docs/multimodal-morning-report.tr.md).

**Ana deney matrisinin 1.808 / 18.460 koşulu tamamlandı (%9,79).** Bu sayı, planlanan 21 koleksiyonu yöntem, aday bütçesi ve rerank K ayarlarıyla genişletir; yinelenen raporları ve küçük teknik kontrolleri dışlar. Ek temsil/parametre ve cevap kalitesi deneyleri bu paydaya dahil değildir. Bu bir kapsam oranıdır; harcanan veya kalan sürenin yüzdesi değildir. [Sayım ve kapsam](reports/multimodal/progress-summary.md).

**Fransızca enerji belgelerinde arama tamamlandı:** 2.225 sayfada 308 sorgu, altı kanalın ve birleşimlerinin iki aday bütçesinde **126 koşulu** ölçüldü. Tek başına ColQwen / EG2 sayfa metni / BM25 Hit@5 değerleri **%86,69 / %82,79 / %77,27**. En yüksek gözlenen birleşim Hit@5 değeri **%88,64**; bu cevap doğruluğu değil, ilgili sayfayı bulma oranı. Bağımsız denetim 38.808 sıralamayı, 659.736 metrik değerini ve 16 eşleştirilmiş güven aralığını yeniden üretti. 26 kaynak grubu üzerinden keşif amaçlı aralıklar hesaplandı. İki eski BM25 koşulu tekrar sayılmadı; ana matrise **124 yeni koşul** eklendi. [Altı kanal, birleşimler ve sınırlar](reports/multimodal/energy-document-retrieval-summary.md).

**ViDoRe V3 bilgisayar bilimi bölümünde görsel belge araması tamamlandı:** 1.360 sayfa üzerinde 215 sorgu, altı kanalın boş olmayan 63 birleşimi ve iki aday bütçesi (**126 koşul**). Tek kanallı sonuçlar:

| Yöntem | Hit@5 | Recall@5 | nDCG@10 |
|---|---:|---:|---:|
| BM25 metin | %92,09 | %53,53 | 0,6334 |
| BGE-M3 metin | %95,35 | %52,45 | 0,6356 |
| EG2 metin | %94,42 | %54,40 | 0,6623 |
| EG2 sayfa görüntüsü | %91,63 | %52,70 | 0,6296 |
| ColQwen2.5 sayfa görüntüsü | %97,21 | %65,34 | 0,7585 |
| EG2 görüntü + metin | %94,42 | %54,54 | 0,6753 |

Hit@5, en az bir doğru sayfa bulmayı; Recall@5, soruya ait bütün doğru sayfaların ne kadarını bulduğumuzu ölçer. ColQwen, 63 ana yöntemde en yüksek gözlenen Recall@5 ve nDCG@10 değerini verdi. BM25 + EG2 sayfa görüntüsü + ColQwen birleşiminin Hit@5 değeri **%98,14** ile en yüksek; ancak Recall@5 değeri tek başına ColQwen’den düşük. Burada reranker veya cevap üretici yok; kaynak bulmayı ölçüyoruz. [Bütün yöntemler](reports/multimodal/vidore-v3-computer_science-en-all-retrieval/report.md) · [Eşleştirilmiş betimsel karşılaştırmalar](reports/multimodal/vidore-v3-computer_science-en-all-retrieval/paired-comparisons.md).

**Belgede iki metin rerankerinin de testleri tamamlandı: 882 koşul.** Aynı 215 sorguda 126 yalnız arama, 378 Laya ve 378 BGE koşulu var; her reranker 63 arama birleşimi × K=20/50/100 × iki bütçede denendi. Ana K=50 koşulunda BGE, 63 eşleşmenin 60'ında nDCG@10'u artırdı; EG2 metnin Hit@5'i **%94,42 → %97,21** oldu. Her yöntemi iyileştirmedi: ColQwen'in nDCG@10'u **0,7585 → 0,7419** düştü. Laya ise 63 ana eşleşmenin tamamında üç ölçümü de düşürdü. Her iki reranker de kaynak sayfa metnini görüyor; tek bağlı kaynak grubu nedeniyle güven aralığı vermiyoruz. [Tam karşılaştırma ve aday sayısı deneyi](reports/multimodal/document-reranker-summary.md).

Aynı 3.600 fotoğraftan oluşan arşivde tam fotoğraf arama ölçümleri:

| Sorgu dili | Sorgu sayısı | EG2 Hit@5 | SigLIP2 Hit@5 | EG2 + SigLIP2 RRF Hit@5 |
|---|---:|---:|---:|---:|
| Türkçe | 7.233 | **%84,64** | %59,88 | %76,28 |
| İngilizce | 7.200 | **%83,75** | %76,89 | %82,92 |

Bunlar bu veri setinde doğru kaynağı bulma oranlarıdır; üretilen cevabın doğruluğu değildir. SigLIP2'nin 64 token girdi sınırı nedeniyle **56 Türkçe sorgu açıkça kısaltıldı**; İngilizce sorgularda kısaltma gerekmedi. Bu fark raporda kayıtlıdır; tablo kısaltmanın tek başına etkisini ölçmez. EG2'nin SigLIP2'ye farkı Türkçede 24,76 yüzde puanı (%95 eşleştirilmiş kaynak grubu güven aralığı: 23,52–26,02), İngilizcede 6,86 puan (5,94–7,82). Eşit ağırlıklı birleşim iki dilde de EG2'yi iyileştirmedi. [Türkçe karşılaştırma](reports/multimodal/xm3600-tr-native-specialist/paired-comparisons.md) · [İngilizce karşılaştırma](reports/multimodal/xm3600-en-native-specialist/paired-comparisons.md).

Sekiz ViDoRe koleksiyonu (**19.252 sayfa / 2.419 ana soru**), Clotho ve Türkçe FLEURS hazır. Görüntü, ses, video ve birleşik girdilerde gerçek yerel model kontrolleri geçti; bunlar başarı oranı ölçümü değildir. Altı kod dilinin tamamı (**183.295 fonksiyon / 52.561 sorgu**) ve 1.000 MSR-VTT videosu hazır. CIRR'nin resmî medyası, yayıncısının erişim süreci tamamlanmadan kullanılamıyor. Aşağıdaki metin RAG pilotu ayrı deney olarak korunuyor.

**Çevresel seste sonuç farklı:** Clotho'nun özgün testinde (1.045 kayıt / 5.225 sorgu) Hit@5, **EG2 ile %11,75, CLAP ile %37,42, birleşimleriyle %26,47**. CLAP farkı 25,67 yüzde puanı (%95 kaynak grubu güven aralığı: 23,25–27,96). Bu yüzden her ortamı ayrı ölçüyoruz. [Ses sonuçları ve eşleştirilmiş karşılaştırma](reports/multimodal/clotho-v2.1-evaluation-native-specialist/paired-comparisons.md). Farklı sorgular ve birden çok doğru kaynak içeren [1.037 sorguluk ek alaka protokolü](reports/multimodal/clotho-dcase2025-additional-relevance-native-specialist/report.md) de tamamlandı.

**Clotho kaynak açıklamaları tamamlandı:** yerel Gemma, yalnız sesi görerek 1.045 kaydın her biri için boş olmayan bir İngilizce açıklama üretti. Sorgular, etiketler ve medya aynı kaldı. 737 farklı açıklama metni var; açıklamaların doğruluğu insan tarafından değerlendirilmedi. Bu galeride 126 arama koşulu tamamlandı: açıklama metninde BM25 / BGE / EG2 Hit@5 **%1,44 / %2,22 / %2,79**; doğrudan seste EG2 / CLAP **%11,75 / %37,42**. EG2 ses + açıklama birleşik girdisi **%6,41** verdi. İki aday bütçesinin tüm 63 altkümesinde en yüksek gözlenen Hit@5 CLAP’te; bu üretilmiş metin temsili aramayı iyileştirmedi. Düşüşün nedeni ve açıklamaların insan değerlendirmesi henüz belirlenmedi. [Tam karşılaştırma ve bağımsız denetim](reports/multimodal/clotho-description-retrieval-summary.md).

**Ek alaka protokolünün açıklama karşılaştırması da tamamlandı:** 126 koşul, 1.037 sorgu ve 3.116 doğru kaynak etiketi aynı 1.045 açıklamayı kullanıyor. Doğrudan EG2 ses / CLAP / EG2 ses+açıklama Hit@5 değerleri **%25,84 / %64,51 / %18,80**; Recall@5 değerleri **%11,32 / %35,08 / %7,07**. Birden çok doğru kayıt olduğundan ilk beşte en az birini bulmak ile doğru kayıtların ne kadarını bulduğumuz farklı ölçüler. Tüm altkümeler ve iki bütçede en yüksek gözlenen Hit@5 CLAP’te. Sorguların %86,50’si tek kaynak grubunda olduğundan güven aralığı vermiyoruz. Altı koşul önceki sonuçlarla aynı; **120 yeni koşul** eklendi. [Ayrı protokol ve bağımsız denetim](reports/multimodal/clotho-additional-description-retrieval-summary.md).

**Türkçe konuşmada tam ASR ve 62 arama koşulu tamamlandı.** Whisper, **743 kaydın tamamını** yazıya çevirdi; **kelime hata oranı %7,29 (13.233 referans kelimede 965 düzenleme)**, boş transkript sayısı sıfır. Aynı 329 transkript sorgusunda Whisper metninde arayan BM25, BGE-M3 ve EG2 ayrı ayrı **Hit@1, Recall@5 ve nDCG@10’da %100** verdi; EG2 ses + Whisper metni birleşik girdisi de aynı sonucu verdi. Yalnız native ses ile **Hit@5 %100 / Recall@5 %99,54**. Sorgular referans transkriptin kendisi olduğundan bu, metin araması için kolay bir kayıt eşleştirme görevi; farklı ifadelerle sorulan sorulardaki başarıyı veya cevap doğruluğunu ölçmüyor. %7,29 hata, yaklaşık her 100 referans kelime için 7,29 ekleme/silme/değiştirme demek; buradan %92,71 anlam doğruluğu çıkaramayız. [ASR, bütün birleşimler ve denetim](reports/multimodal/speech-asr-summary.md).

**Türkçe konuşmanın yeniden sıralaması tamamlandı: 434 koşul** (62 yalnız arama + 186 Laya + 186 BGE), aynı 329 sorgu ve 743 kayıt üzerinde ölçüldü. Ana K=50 koşulunda BGE, 31 arama altkümesinin tamamında Hit@5’i korudu; Recall@5 üçünde arttı, 28’inde aynı kaldı. Laya, 31 altkümenin tamamında Recall@5 ve nDCG@10’u düşürdü; Hit@5 ise 25’inde düştü, altısında değişmedi. Native sesin Recall@5’i BGE ile **%99,54’ten %100’e** çıktı. İki reranker de yalnız Whisper metnini görüyor. Bu sonuçlar transkriptle kayıt eşleştirmesidir; üretilen cevabın doğruluğu değildir. [Tam tablo, bağımsız denetim ve 93 eşleştirilmiş karşılaştırma](reports/multimodal/speech-reranker-summary.md).

**Video araması tamamlandı:** MSR-VTT 1K-A üzerinde (1.000 sorgu / 1.000 video) Hit@5, **EG2 ile %75,00, CLIP kare ortalamasıyla %53,80, birleşimleriyle %67,10**. EG2 bu CLIP koşulunun 21,20 yüzde puanı önünde (%95 eşleştirilmiş güven aralığı: 18,20–24,20). İkisi de saniyede bir, en fazla 16 görüntü karesi kullanıyor; bu koşulda videonun ses kanalı işlenmiyor. [Video karşılaştırması](reports/multimodal/msrvtt-1k-a-native-specialist/paired-comparisons.md).

**BM25, altı kod ve sekiz belge koleksiyonunun tamamında 54.980 sorguyu bitirdi.** Yalnız pozitif kelime eşleşmeleri sonuçlara alındı; kalan embedding, birleşim ve reranker koşulları bekliyor. [Başlangıç ölçümleri ve kesin sayılar](reports/multimodal/bm25-summary.md).

**Altı kod dilinin arama karşılaştırması tamamlandı:** Python, Ruby, Go, Java, PHP ve JavaScript; ayrı galerilerde toplam 183.295 fonksiyon ve 52.561 sorgu. Yedi yöntem × iki aday bütçesi × altı dil ile **84 koşul** ölçüldü. Kanal başına aday bütçesi kullanılan ana koşulda Hit@5 sonuçları:

| Yöntem | Python Hit@5 | Ruby Hit@5 | Go Hit@5 | Java Hit@5 | PHP Hit@5 | JavaScript* Hit@5 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | %38,93 | %45,60 | %61,94 | %39,42 | %33,27 | %36,62 |
| BGE-M3 | %62,18 | %68,20 | %87,70 | %63,73 | %57,74 | %58,22 |
| EmbeddingGemma 2 | %84,46 | %86,28 | %96,28 | %83,80 | %75,75 | %79,91 |
| BM25 + BGE-M3 | %59,41 | %64,71 | %85,71 | %61,36 | %54,10 | %56,40 |
| BM25 + EG2 | %69,25 | %70,74 | %89,51 | %71,11 | %62,58 | %64,21 |
| BGE-M3 + EG2 | %76,77 | %78,75 | %93,39 | %76,99 | %70,81 | %71,25 |
| BM25 + BGE-M3 + EG2 | %73,56 | %74,94 | %92,07 | %74,66 | %67,48 | %68,55 |

*JavaScript ayrı bir girdi protokolü kullanır: iki uzun fonksiyon iki encoder için aynı kayıpsız sınırlarda parçalanır; 13.981 fonksiyon 14.084 parçayla temsil edilir ve fonksiyon puanı en yüksek parça benzerliğidir. Diğer beş dil bütün fonksiyonları embed eder.

Bu altı sabit kod arama testinde EG2 önde; eşit ağırlıklı RRF birleşimi onu iyileştirmedi. Kaynak yorumları ve referans docstring çıkarıldı; burada reranker veya cevap modeli kullanılmadı. [İki aday bütçesi, MRR/nDCG ve eşleştirilmiş karşılaştırmalar](reports/multimodal/code-dense-summary.md).

**Koda özel sorgu öneki altı dilde ayrıca denendi.** Belge vektörleri aynı tutulduğunda, genel önek → kod öneki Hit@5 değişimleri: Python **%84,46 → %85,07**, Ruby **%86,28 → %86,12**, Go **%96,28 → %96,90**, Java **%83,80 → %84,49**, PHP **%75,75 → %75,65**, JavaScript **%79,91 → %80,46**. Dört dilde artış, ikisinde düşüş gözlendi; güven aralığı hesaplanmayan bu farkları kesin iyileşme diye yorumlamıyoruz. 52.561 sıralama ve 893.537 metrik değeri bağımsız doğrulandı. [Ayrı önek karşılaştırması](reports/ablations/code-query-prefix/README.md).

**Sekiz native veri görünümünde boyut taraması tamamlandı:** 120 boyut/yöntem/aday bütçesi koşulu, önce özgün 768 boyut sonuçları doğrulanarak hesaplandı. Türkçe fotoğraf Hit@5, **512 boyutta %84,54; 768 boyutta %84,64**. Ham N-vektör alanı üçte bir azalıyor. 128 boyutta oran %66,65'e düşüyor. Videoda Hit@5, 512 boyutta %74,70; 768 boyutta %75,00. Bunlar bu veri kümelerinde gözlenen değerlerdir; çıkarım hızı veya istatistiksel eşdeğerlik iddiası değildir. [Boyut sonuçları](reports/multimodal-dimensions/README.md).

**Gemma ilgililik kontrolü geçti:** sabit beş sorguda 63 arama başlangıç koşulu ve K=20 ile 63 Gemma yeniden sıralama koşulu tamamlandı; 278 farklı sorgu–sayfa çifti yeni puanlandı. Bu, yerel modelin ve skor önbelleğinin çalıştığını doğrular; tam veri üzerinde sıralama doğruluğu ölçümü değildir. [Teknik kontrol](reports/multimodal/vidore-v3-computer_science-en-gemma-technical-5q/report.md).

**Belge cevap değerlendirmesinin veri hazırlığı doğrulandı:** sekiz sabit ViDoRe koleksiyonundaki 2.419 sorunun tamamı, yayımlanan referans cevap ve doğru kaynak etiketleriyle eşleşiyor. Yeni hazırlık modülü referansları yalnız değerlendirmede kullanıyor; cevaptaki kelime örtüşmesi ile kaynak atıflarının eşleşmesini ayrı hesaplıyor. Yerel cevap üretme çalıştırıcısı, sabit kanıt ve önbellek denetimleriyle geliştirildi ve CPU testlerini geçti; tam cevap kalitesi deneyi henüz yapılmadı; altı tek soruluk teknik denemenin dördü geçti, ikisi geçersiz kaynak atıfları nedeniyle reddedildi. İş planlayıcısı, kontrol cevaplarını koleksiyon başına bir kez; gerçek kaynaklı cevapları tamamlanan her arama kimliği için planlıyor. Kaynak ve rapor hash’leri üretimden önce denetleniyor. Henüz tam QA veya anlam doğruluğu ölçümü yok. [QA çalışma kapsamı](reports/multimodal-qa-execution-plan-summary.json) · [Hazırlık denetimi](reports/multimodal-qa-readiness.json) · [Değerlendirme kurulumu ve sınırlar](docs/multimodal-running.md#document-answer-evaluation-preparation).

**Üç ters arama protokolü tamamlandı:** görüntüden referans açıklama aramada Hit@5 **Türkçede %87,42, İngilizcede %89,14**; aynı 3.600 görüntü, ayrı 7.233 / 7.200 metinlik galeriler kullanıldı. Sesten normalize referans transkript aramada, 743 kayıt / 329 metin üzerinde **Hit@5 %100**. Mevcut medya vektörleri doğrulanarak yeniden kullanıldı; yalnız metin galerileri yeni embed edildi. Bağımsız denetim 7.943 sıralama ve 135.031 metrik değerini doğruladı. Galeride referans metinler bilerek bulunuyor (`gold_fields_used=true`); bu kaynak üzerinden ASR üretimi veya serbest cevap doğruluğu değil. Bu üç ek deney ana matris sayısına dahil değil. [Sonuçlar ve veri kaynağı](reports/ablations/reverse/README.md).

Whisper'ın Transformers önbellek uyumluluğu düzeltmesi, regresyon testlerini ve 16,92 / 36,84 saniyelik kayıtlarda gerçek [CPU](reports/multimodal-whisper-cpu-preflight.json) ve [MPS](reports/multimodal-whisper-mps-preflight.json) kontrollerini tüm ses örneklerini koruyarak geçti. Sonraki model işleri macOS Metal derleyici servisine erişim hatasıyla kesildi; kurtarma sırasında yeni GPU süreci ve iki gerçek Whisper kontrolü geçti. Kurtarılan tam ASR koşusu ve arama karşılaştırması şimdi tamamlandı; sıradaki diğer deneyler henüz bitmedi.

[Ölçülen bütün koşular](reports/multimodal/README.md) · [Yerelde çalıştırma](docs/multimodal-running.md) · [Güncel çalışma durumu](docs/multimodal-status.md) · [Ayrıntılı deney planı](docs/multimodal-plan.tr.md) · [Planlanan bütün varyantlar](configs/multimodal-variants.csv)

## Sistem ne yapıyor?

RAG, cevap verecek modele soruyla ilgili kaynak metinleri önceden sunmaktır. Örneğin kullanıcı “Bu kurum ne zaman kuruldu?” diye sorar:

1. **Arama:** sistem bütün kaynaklar arasından ilgili olabilecek en fazla 50 metin getirir.
2. **Yeniden sıralama:** Laya açıksa bu adayları soru açısından değerlendirir; en yüksek puanlı en fazla 5 metin seçilir. Kapalıysa aramanın ilk 5 sonucu kullanılır.
3. **Cevap:** aynı yerel Gemma 4 modeli soruyu ve bu 5 metni alıp kaynak kimlikleriyle cevap yazar.

BM25 kelime örtüşmesini, embedding modeli anlam yakınlığını ölçer. Hibrit kolda iki arama paralel yapılır, sonuç sıraları birleştirilir. **Bu RAGTurk pilotunda BGE-M3 embedding modelidir; pilotta ayrıca BGE reranker kullanılmaz.** Yukarıdaki çoklu ortam paketi, BGE reranker koşullarını ayrıca içerir.

Laya, TypeSafe'ın Jev modelinin resmî açık kaynak sürümü değildir. **Convai Innovations'ın geliştirdiği bağımsız, açık kaynaklı bir alternatiftir.** Bu projede son cevabı yazmaz; kaynak metinleri puanlar. [Laya kaynak kodu](https://github.com/NandhaKishorM/laya), [Türkçe için kullanılan multilingual model](https://huggingface.co/convaiinnovations/laya-multilingual).

## Hangi yollar karşılaştırılıyor?

| Arama yöntemi | Laya kapalı | Laya açık |
|---|---|---|
| BM25 | İlk 5 kaynak → Gemma 4 | İlk 50 → Laya → 5 kaynak → Gemma 4 |
| BGE-M3 | Aynı akış | Aynı akış |
| EmbeddingGemma 2 | Aynı akış | Aynı akış |
| BM25 + BGE-M3 | Aynı akış | Aynı akış |
| BM25 + EmbeddingGemma 2 | Aynı akış | Aynı akış |

Toplam **10 deney kolu** var. Hepsinde aynı Gemma 4 26B-A4B, aynı soru kümesi, aynı kaynak havuzu ve aynı cevap üretim ayarları kullanılır. Arama yöntemlerinin bulduğu adaylar farklı olabilir; bu fark zaten ölçmek istediğimiz şeydir. Laya açık/kapalı karşılaştırmasında ise **aynı arama kolunun aynı adayları** kullanılır. BM25'te pozitif eşleşme puanlı 50 metin bulunmazsa aday sayısı daha az olabilir; liste ilgisiz metinlerle tamamlanmaz.

## Veri ve ölçek

Ana veri kaynağı [METU NLP RAGTurk](https://huggingface.co/datasets/metunlp/ragturk). Türkçe Wikipedia ve CulturaX metinlerinden oluşturulan kaynaklar, sentetik soru–cevaplar ve kaynak ilişkileri içerir. [Makale](https://aclanthology.org/2026.sigturk-1.15/).

Makale JSON'ları ile yayımlanmış web metinleri, parça konumları ve soru dosyaları birleştirildi. Sabitlenen indirilebilir sürümden **8.104 kullanılabilir makale, 37.511 parça ve 14.530 soru** elde edildi. Makalenin **58.289 parça / 20.459 soru** sayıları bu indirme ile karşılanmıyor. Eksik kayıt üretilmedi; sırf 100 bin parçaya ulaşmak için metinler çoğaltılmadı.

**2.000 soru geliştirmeye, 12.530 soru son teste** ayrıldı; seed 42. Aynı makale, normalize edildiğinde birebir aynı soru veya aynı doğru kaynak metni iki bölüme dağıtılmaz. Anlamca benzer ama birebir aynı olmayan tekrarların tamamen temizlendiği iddia edilmez. Bölümleme bu projeye aittir; resmî RAGTurk test bölümü değildir. On varyantın tam testi **125.300 soru–varyant çıktısı** demektir; aynı cevap isteği önbellekten kullanılabildiğinde gerçek üretim çağrısı daha az olabilir.

Dört bozuk soru satırı ve yardımcı dosyası eksik bir makale dışarıda bırakıldı. Soru dosyası bulunmayan sekiz makalenin geçerli metinleri arama havuzunda tutuldu. Bazı üst veriler makaleyle çelişiyor; bunlar gizlenmeden manifestte kaydedilir. Soru–cevaplar ve kaynak etiketleri sentetiktir; bağımsız insan denetiminden geçmiş kusursuz referanslar sayılmaz. Denetim ayrıntıları [sonuçlar](docs/results.md) sayfasında ve yerelde oluşan `data/ragturk/manifest.json` dosyasında.

## Çalıştırma

İlk tam deney için hedef donanım **M4 Max / 128 GB birleşik bellek**. Bu, ölçüm yapılacak makinedir; asgari sistem gereksinimi veya hız garantisi değildir.

Python 3.11–3.13 ve `uv` ile, depo dizininden:

```bash
uv sync --extra models --locked
uv run --locked --extra models rag-benchmark doctor --config configs/ragturk.toml
uv run --locked --extra models rag-benchmark prepare --config configs/ragturk.toml
uv run --locked --extra models rag-benchmark prepare-models --config configs/ragturk.toml --only bge,embeddinggemma,laya,generator --include-runtime
```

Gemma sunucusunu **ayrı bir terminalde** başlatıp açık bırakın:

```bash
uv run --locked --extra models rag-benchmark serve --config configs/ragturk.toml
```

İlk terminalde küçük geliştirme deneyi:

```bash
uv run --locked --extra models rag-benchmark run --config configs/ragturk.toml --split dev --limit 20 --variants bm25,bm25_laya --run-dir runs/pilot
uv run --locked --extra models rag-benchmark report --run-dir runs/pilot
uv run --locked --extra models rag-benchmark export-report --run-dir runs/pilot --output-dir reports/pilot
```

`export-report`, yeni bir dizine `report.md`, `summary.json`, `provenance.json` ve `per-query-metrics.jsonl` yazar. Yayımlanacak bu dosyalarda seçilmiş ayarlar, kimlikler ve ölçümler bulunur; sorular, cevaplar, kaynak metinler, promptlar ve yerel dosya yolları çıkarılır. Var olan çıktı dizininin üzerine yazılmaz.

`serve`, GGUF dosyasının SHA-256 özetini doğrular; llama.cpp'yi Metal ile `http://127.0.0.1:8080/v1` adresinde, `gemma4` adıyla çalıştırır. Bu yerel protokol, OpenAI hizmeti veya API anahtarı kullanmaz. GGUF dosyası ve doğrulama özeti [`configs/ragturk.toml`](configs/ragturk.toml) içindedir. Güncel cevap üretim ayarları tüm kollarda sıcaklık 0, seed 42, en fazla 512 çıktı tokenı ve düşünme modu kapalıdır. Önceki pilotun 256 token sınırı yorum cevaplarını kestiği için ortak sınır geliştirme aşamasında artırıldı. Teknik ayrıntılar [model sözleşmeleri](docs/models.md) sayfasında.

Yalnız aramayı sınamak için `run` komutuna `--retrieval-only` eklenebilir; bu mod cevap üretmediğinden uçtan uca RAG sonucu değildir. Hazırlanan veriler `data/ragturk/` altında tutulur. Yöntem kimlikleri `bm25`, `bge`, `embeddinggemma`, `bm25_bge`, `bm25_embeddinggemma`; Laya'lı kola `_laya` eklenir.

**On varyantın tamamını 20 geliştirme sorusunda** çalıştırmak için `--variants` seçeneğini kullanmayın; ayar dosyasındaki bütün kollar çalışır:

```bash
uv run --locked --extra models rag-benchmark run --config configs/ragturk.toml --split dev --limit 20 --run-dir runs/matrix-dev-pilot
```

Pilotu kontrol ettikten sonra **2.000 geliştirme sorusu × on varyant** için `--limit` seçeneğini kaldırın ve yeni bir çıktı dizini kullanın:

```bash
uv run --locked --extra models rag-benchmark run --config configs/ragturk.toml --split dev --run-dir runs/matrix-dev-full
```

`--split test`, bütün ayarlar sabitlendikten sonra 12.530 son test sorusu için, ayrı bir deney diziniyle kullanılır. Prompt, birleştirme veya çıktı bütçesi ayarları geliştirme verisiyle seçilir.

Önce 20 soruluk geliştirme kontrolü yapılır. Tam deney bundan sonra çalıştırılır. İlk veri/model indirmeleri internet gerektirir; gerekli dosyalar hazır olduğunda temel RAG deneyi yerel çalışır. Aşağıdaki isteğe bağlı Jev değerlendirmesi ayrı bir hosted API adımıdır. Yerelde çalışmak donanım, bellek ve elektrik maliyetini ortadan kaldırmaz.

## Sonuçları nasıl okumalı?

- **Recall@50:** ilgili diye etiketlenmiş kaynakların ne kadarı ilk 50 adaya girdi? Buraya girmeyen kaynağı Laya geri getiremez.
- **Recall@5:** ilgili kaynakların ne kadarı son seçilen en fazla 5 metin içinde?
- **MRR/nDCG:** doğru kaynak sıralamanın ne kadar başında?
- **Cevap EM/F1:** üretilen cevap, referans cevabın kelimeleriyle ne kadar örtüşüyor?
- **Gecikme:** arama, Laya ve cevap üretimi ayrı ayrı ne kadar sürüyor?

**%99 kaynak bulma, %99 doğru cevap anlamına gelmez.** Model doğru kaynağı görüp yanlış cevap verebilir. Doğru bir cevabın farklı kelimelerle yazılması da F1'i düşürebilir; bu metrik tek başına insanın yaptığı olgusal doğruluk denetimi değildir.

Laya'nın verdiği 0–1 puanı bu görev için doğrulanmış bir güven yüzdesi değildir. İlk deneyde düşük puanlı her metni otomatik silen eşik kullanılmaz; puana göre ilk 5 seçilir. Eşik, prompt veya model ayarı gerekiyorsa geliştirme verisinde seçilir; son test sonuçlarına bakılarak ayar değiştirilmez.

Önbellekteki aynı cevabı yeniden kullanmak zaman kazandırabilir. Token sınırında durma bilgisi önbellekte korunur; önbellekten okuma süresi yeni cevap üretme hızına dahil edilmez. Deneyde soru embedding'leri yeniden hesaplanır. Hazırlık sonrası ilk soru ve ilk gerçek cevap üretimi ilgili gecikme özetlerinden çıkarılır. Sunucunun bildirdiği derleme ve çalışma ayarları deney/önbellek kimliğine katılır.

## İsteğe bağlı anlamsal değerlendirme

[Jev değerlendirme akışı](docs/semantic-evaluation.md), kaydedilmiş cevapları iki ayrı görevde inceler: referans ve doğru kaynaklara göre **cevabın anlamca doğruluğu**; yalnızca Gemma'ya gerçekten verilmiş metinlere göre **cevabın kaynaklarla desteklenmesi**. Yerel RAG deneyi sonrasında OpenRouter'ın native Decisions API'si, `typesafe/jev-1.13` modeli ve `OPENROUTER_API_KEY` kullanılır. Doğrulanan servis sürümü `typesafe/jev-1.13-20260917`'dir. Jev bu adımda Laya'nın yerini almaz.

[Jev pilotu](reports/semantic-pilot-openrouter/report.md) tamamlandı: **20 farklı sorudan 200 cevap**, 400 değerlendirme görevinin tekrarları birleştirilince **319/319 doğrulanmış istek**. Laya kapalı EmbeddingGemma 2 ve BM25+EmbeddingGemma 2 kollarının ikisinde de **14/20 cevap `correct` etiketi aldı (%70)**. Kontroller hariç değerlendirme maliyeti **0,031974726 ABD doları** oldu. Bu, insanın doğruladığı kesin doğruluk oranı değildir: incelemede Jev'in kaynakla açık çelişen bir donanım bilgisini desteklenmiş saydığı görüldü; [sentetik kontrollerde](reports/semantic-controls-tr/report.md) de hatalar vardı. Türkçe insan kalibrasyonu henüz yapılmadı; 12.530 soruluk son test değerlendirilmedi. [Ayrıntılı yorum](docs/semantic-evaluation.md).

## Jev ile GPT 6.1 Sol Ultra karşılaştırması

[Aynı 200 cevabın ikinci hakem değerlendirmesi](reports/semantic-judge-comparison/analysis.tr.md) tamamlandı. `gpt-6.1-sol` ve `ultra` ayarıyla çalışan üç paralel Codex ajanı, Jev etiketlerini ve yöntem adlarını görmeden aynı kuralları uyguladı. Doğruluk etiketlerinde uyum **150/200 (%75)**, kaynak desteğinde **166/200 (%83)** oldu. Tekrarsız görevlerde doğruluk uyumu **83/130 (%63,85)**; cevaplar yalnızca 20 farklı soruya aittir.

BM25 + EmbeddingGemma 2, Laya kapalı yolda iki hakem de **14/20 (%70)** tam doğru dedi; ancak **yalnızca 11 cevabı ortak olarak tam doğru** saydılar. Sol 14 doğru, 4 kısmi ve 2 değerlendirilemez etiket verdi; hem doğru hem kaynakla destekli saydığı cevap **13/20 (%65)**. Bu, değişmeyen Gemma cevaplarına verilen farklı notlardır. İnsan değerlendirmesi henüz yapılmadı; hakemlerin aynı notu vermesi veya birinin daha yüksek toplam puan vermesi doğruluğunu kanıtlamaz. Sol ajanları gruplu ve kalıcı bağlamla çalıştığı için yöntem, izole Jev API çağrılarıyla birebir aynı çalışma düzeni değildir.

## İlk ölçümler

[On varyantlı pilotta](reports/matrix-dev-pilot-512/report.md), **Laya kapalı BM25 + EmbeddingGemma 2** en yüksek gözlenen Recall@5 (**0,9500**) ve cevap F1'ini (**0,5326**) verdi. BM25 + BGE-M3 bu iki ölçümde yakındı; nDCG@10 değeri en yüksekti. Laya, bu 20 soruda beş arama kolunun tamamında Recall@5 ve F1'i düşürdü. Küçük bir geliştirme örnekleminden nihai kazanan çıkarılamaz.

200 çıktının 3'ü ortak 512 token sınırına ulaştı ve sonuçlardan çıkarılmadı. 11 çıktı aynı üretim isteğinin önbelleğinden geldi. Promptun uygunluğu ve çıktı bütçesi, son ayarlar sabitlenmeden önce daha geniş geliştirme verisinde değerlendirilmeli. [2.000 soruluk BM25 sonucu](reports/bm25-dev-2000/report.md) ve önceki iki kollu pilotlar [sonuçlar](docs/results.md) sayfasında korunuyor; son test verisine geçilmedi.

## Lisans

Projenin kodu **MIT** lisanslıdır. RAGTurk veri yayını **CC BY-NC-SA 4.0** olarak etiketlenmiştir; model ağırlıkları da kendi koşullarını korur. Kodun açık kaynak olması, veri ve modelleri otomatik olarak sınırsız ticari kullanıma açmaz. Ham veri ve model ağırlıkları bu depoda yayımlanmaz.

Bu bağımsız bir değerlendirme projesidir. Makaledeki veya önceki Jev benchmark'ındaki sonuçlar, bu yeni deney yapılmış gibi gösterilmez. Ayrıntılar için [protokol](docs/protocol.md) ve [sonuç durumu](docs/results.md).
