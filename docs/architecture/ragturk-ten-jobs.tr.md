# RAGTurk: on bağımsız iş, pilottan tam koşuya

Kullanıcının istediği çalışma birimi **senaryodur**: beş arama yöntemi × Laya açık/kapalı = on ayrı iş. Her işin pilotu, hataları, ilerlemesi, tam koşusu ve sonucu ayrı izlenir. Ortak hazırlık yeniden yapılmaz. Teknik olarak doğrulanan senaryo tam geliştirme kümesine geçer; yüksek doğruluk almak geçiş koşulu değildir.

[On işlik plan](../../configs/examples/ragturk-ten-jobs.proposal.json) henüz mevcut CLI tarafından yürütülmez; dağıtık scheduler veya yeni model servisi uygulanmış değildir. Bu değişiklikte yalnız mevcut kayıtlar CPU ile denetlendi. GPU, API, cloud tahsisi veya eski kuyruk başlatılmadı. Önceki multimodal işler ve otomasyon ayrı olarak durdurulmuş durumda kalır.

## Gerçek pilot denetimi

[Bağımsız denetim](../../reports/ragturk-parallel-pilot-audit.json), 6 Ekim pilotunun özel SQLite/prediction kayıtları ile yayımlanan export/summary/provenance dosyalarını karşılaştırdı. Girdilerin denetim boyunca değişmediği kontrol edildi. Her iş aynı 20 benzersiz sorguyu ve 37.511 parçalık galeriyi kullanır; 50 benzersiz aday, ilk beş bağlam ve Laya çiftlerinde aynı aday kimlikleri/skorları doğrulandı. Eksik veya hata satırı yok. Bu, bugünkü ortamın sağlık testi değildir.

| İş | Arama | Reranker | Kayıt | Normal bitiş | Token sınırı | Tam geliştirme |
|---|---|---|---:|---:|---:|---|
| 01 | BM25 | Yok | 20/20 | 20 | 0 | Başlamadı |
| 02 | BM25 | Laya | 20/20 | 20 | 0 | Başlamadı |
| 03 | BGE-M3 | Yok | 20/20 | 19 | 1 | Başlamadı |
| 04 | BGE-M3 | Laya | 20/20 | 20 | 0 | Başlamadı |
| 05 | EmbeddingGemma 2 | Yok | 20/20 | 19 | 1 | Başlamadı |
| 06 | EmbeddingGemma 2 | Laya | 20/20 | 20 | 0 | Başlamadı |
| 07 | BM25 + BGE-M3 | Yok | 20/20 | 19 | 1 | Başlamadı |
| 08 | BM25 + BGE-M3 | Laya | 20/20 | 20 | 0 | Başlamadı |
| 09 | BM25 + EmbeddingGemma 2 | Yok | 20/20 | 20 | 0 | Başlamadı |
| 10 | BM25 + EmbeddingGemma 2 | Laya | 20/20 | 20 | 0 | Başlamadı |

Toplam 200 kayıt: 197 normal bitiş, üç token sınırı, 11 aynı-istek cache kullanımı, 189 tarihsel yeni üretim. Üç sınırlı çıktı bütün eski metriklerde kalır. Bu denetim yeni üretim yapmadı; normal bitiş cevap doğruluğu veya insan kalibrasyonu değildir.

## İstenen yürütme düzeni

1. **Ortak hazırlığı doğrula.** Donmuş dataset/split ve mevcut BM25, BGE, EmbeddingGemma belge indekslerini doğrula. Eksik artifact varsa tek sahibi üretir; belge vektörleri on kez çıkarılmaz.
2. **On ayrı senaryo işi oluştur.** Her işin kimliği ve sonuç klasörü ayrıdır. Beş arama sonucunun her biri kendi Laya açık/kapalı çiftine aynı aday listesini verir. Uyumlu BGE/EmbeddingGemma sorgu vektörleri de ortak aşamalardır.
3. **Pilotu ayrı ayrı doğrula.** Aynı sorular, kaynak kapsamı, aday bütçesi, model/runtime, cevap bitişi ve hata kayıtlarını denetle. Donanım/serving değiştiyse sınırlı gerçek runtime pilotu gerekir; tarihsel cevabı cache'den okumak yeterli değildir.
4. **Hazır işi tam geliştirmeye geçir.** Her iş aynı 2.000 soruyu işler: toplam 20.000 soru–senaryo kaydı. Pilot ve tam koşu farklı klasör/kimlik kullanır. Bir işin gecikmesi bağımsız hazır işleri bekletmez; ortak protokol/servis hatası ona bağlı işleri durdurur.
5. **Doğrulayarak birleştir.** Kapsam, tekrarlar, protokol ve gerçek runtime eşleşir. On ayrı skor ve beş Laya karşılaştırması korunur. Kapanmış checkpoint kalıcı depoya alınır; bağımsız denetim sonrası güvenli sonuç yayımlanır.
6. **Nihai testi ayrı tut.** Geliştirme kararları dondurulduktan sonra 12.530 soruluk test aynı on işe verilebilir. Bu 125.300 kayıtlık ayrı aşamadır; geliştirme sırasında otomatik açılmaz.

Üç token-sınırı gözlemi için tam koşudan önce ortak bitiş/bütçe politikası belirlenir. Sabit 512 sınırını koruyup sınırlı cevapları sonuç ve hata oranlarıyla raporlamak mümkündür. Daha geniş bütçe seçilirse bütün karşılaştırmaya uygulanır ve geliştirme verisinde yeni kimlikle doğrulanır. Yalnız düşük skor alan senaryoya daha büyük bütçe verilmez; sınırlı cevaplar paydadan çıkarılmaz. Düşük accuracy, doğru çalışan bir deneyin geçerli sonucu olabilir; yalnız iyi modelleri tam teste geçirmek seçilim yanlılığı yaratır.

## Ajan, iş ve GPU ayrımı

On senaryonun yönetimi bağımsız olabilir. **Ajan sayısı GPU sayısı veya eşzamanlı model kapasitesi değildir.** Plan on işi izler; hesaplama worker'ları uygun işleri kapasiteye göre alır. Aynı GPU'ya on ayrı model kopyası yüklemek gerekli değildir.

| Kaynak düzeni | On iş nasıl ilerler? |
|---|---|
| Tek GPU | On mantıksal iş; model sahibi servis ölçülmüş batch/in-flight sınırıyla çalışır. CPU hazırlığı ve değerlendirme paralel olabilir. On kat hız vaat edilmez. |
| İki uygun GPU/worker | Uyumlu iki ağır iş aynı anda çalışabilir; biri bitince sıradaki atanır. Ortak artifact'ler doğrulanarak taşınır. |
| On uygun GPU/worker | Her senaryo ayrı worker'a atanabilir; aynı donmuş koşullar gerekir. Ortak servis veya sorgu shard'ları daha verimli olabilir. |
| Karışık Mac/CUDA/API havuzu | Atama ve sayısal koşullar sürümlenir; CUDA çıktısı eski MPS baseline'ı gibi etiketlenmez. API kotası ayrı kapasitedir. |

Güçlü GPU'da batch veya aynı modelin birden çok isteği birlikte işlemesi hız kazandırabilir. Kaç istek kaldırdığı ölçülür; hız çarpanı GPU adına veya ajan sayısına eşit kabul edilmez. Yarışan işlerin zaman ölçümlerine etkisi ayrıca belirtilir.

## Bugünkü kod ve eksikler

| Alan | Hazır olan | Gereken |
|---|---|---|
| Senaryo seçimi | `--variants` ile tek senaryo, ayrı `--run-dir` | On işin yaşam döngüsü coordinator'ı |
| Sonuç yazımı | Aynı run-directory için gerçek kilit | GPU/worker sahipliği, durma ve kalıcı checkpoint |
| Belge indeksi | Cache yeniden kullanımı | Soğuk build başına tek producer; BM25 çoklu dosyalarına eşzamanlı yazma engeli |
| Sorgu/arama | Runner sorgu cache'ini kapalı çağırır | Ortak query/candidate artifact'leri; hesap tekrarını kaldırma |
| Cevap cache'i | SQLite WAL işlemleri | miss→generate→save için atomik sahiplik; aynı isteğin çift üretimini engelleme |
| Model yükleme | Her runner SDK embedding/Laya kopyası yükleyebilir | Ölçülmüş kapasiteyle admission ve ortak servis/adaptör |
| Gemma | Mevcut `serve`, `--parallel 1` kullanır | Doğrulanmış serving concurrency ve açık runtime kimliği |
| Paired analiz | Aynı run-directory içindeki çiftleri karşılaştırır | Ayrı işlerden kimlik/kapsam kontrollü merge ve beş paired analiz |

Mevcut komutu on terminalde sınırsız başlatmak bu düzeni sağlamaz. Uygulama sırası: ortak aşamalar ve cache sahipliği → ayrı iş/merge doğrulaması → gerçek runtime pilotu → kaynaklara göre paralel tam geliştirme koşusu. Tarihsel pilot kalitesi ile yeni yürütücünün doğruluğu ayrı doğrulanır. Mevcut kayıtlar ve protokol kimlikleri korunur.

Kaynaklar: [CLI](../../src/rag_benchmark/cli.py), [runner](../../src/rag_benchmark/runner.py), [retrieval](../../src/rag_benchmark/retrieval.py), [storage](../../src/rag_benchmark/storage.py), [pilot sonuçları](../results.md), [genel mimari](modular-benchmark.tr.md).
