# rag-benchmark

**Türkçe RAG sistemlerini aynı koşullarda, yerel modellerle karşılaştırmak için bir deney düzeneği.** BM25, BGE-M3 ve EmbeddingGemma 2 belge aramasını; Laya'nın yeniden sıralamaya katkısını; aynı Gemma 4 modelinin ürettiği cevapları ölçer.

[English](README.md) · [Deney protokolü](docs/protocol.md) · [Sonuçlar ve doğrulama durumu](docs/results.md) · [Anlamsal değerlendirme](docs/semantic-evaluation.md)

**Mevcut durum:** **20 soru × on varyant pilotu tamamlandı: 200/200 çıktı**, 37.511 parçanın tamamında arama yapıldı. İki dense indeks de hazır. Ayrı 2.000 soruluk BM25 arama deneyi tamamlandı; on varyantın 2.000 soruluk geliştirme deneyi ve 12.530 soruluk son test henüz çalıştırılmadı. [CI](https://github.com/erendikmenn/rag-benchmark/actions/workflows/ci.yml) · [Ölçümler ve sınırları](docs/results.md).

## Çoklu ortam benchmark'ı — tam veri koşuları başladı

Yeni paket **fotoğraf, görsel belge, çevresel ses, Türkçe konuşma, video, kod ve görüntü+talimat sorgularını** ölçüyor. Kayıtlı kapsam **321 arama kombinasyonu × dört sıralama koşulu = 1.284 yöntem ailesi**; veri/dil bölümleri ve ek ayarlar bunun üzerine geliyor. Bunlar planlanan yöntemlerdir; **tamamlanmış deney sayısı değildir**.

İlk tam bölüm ölçümü hazır: **ViDoRe V3 bilgisayar bilimi, 215 soru / 1.360 sayfa üzerinde BM25: Hit@5 %92,09, Recall@5 %53,53, nDCG@10 0,6334**. Hit, en az bir doğru sayfa bulmayı; recall, soruya ait bütün doğru sayfaların ne kadarını bulduğumuzu ölçer. Bunlar cevap doğruluğu yüzdeleri değildir. [Ölçüm raporu](reports/multimodal/vidore-v3-computer_science-en-bm25/report.md).

Aynı 3.600 fotoğraftan oluşan arşivde tam fotoğraf arama ölçümleri:

| Sorgu dili | Sorgu sayısı | EG2 Hit@5 | SigLIP2 Hit@5 | EG2 + SigLIP2 RRF Hit@5 |
|---|---:|---:|---:|---:|
| Türkçe | 7.233 | **%84,64** | %59,88 | %76,28 |
| İngilizce | 7.200 | **%83,75** | %76,89 | %82,92 |

Bunlar bu veri setinde doğru kaynağı bulma oranlarıdır; üretilen cevabın doğruluğu değildir. SigLIP2'nin 64 token girdi sınırı nedeniyle **56 Türkçe sorgu açıkça kısaltıldı**; İngilizce sorgularda kısaltma gerekmedi. Bu fark raporda kayıtlıdır; tablo kısaltmanın tek başına etkisini ölçmez. EG2'nin SigLIP2'ye farkı Türkçede 24,76 yüzde puanı (%95 eşleştirilmiş kaynak grubu güven aralığı: 23,52–26,02), İngilizcede 6,86 puan (5,94–7,82). Eşit ağırlıklı birleşim iki dilde de EG2'yi iyileştirmedi. [Türkçe karşılaştırma](reports/multimodal/xm3600-tr-native-specialist/paired-comparisons.md) · [İngilizce karşılaştırma](reports/multimodal/xm3600-en-native-specialist/paired-comparisons.md).

Sekiz ViDoRe koleksiyonu (**19.252 sayfa / 2.419 ana soru**), Clotho ve Türkçe FLEURS hazır. Görüntü, ses, video ve birleşik girdilerde gerçek yerel model kontrolleri geçti; bunlar başarı oranı ölçümü değildir. Altı kod dilinin tamamı (**183.295 fonksiyon / 52.561 sorgu**) ve 1.000 MSR-VTT videosu hazır. CIRR'nin resmî medyası, yayıncısının erişim süreci tamamlanmadan kullanılamıyor. Aşağıdaki metin RAG pilotu ayrı deney olarak korunuyor.

**Çevresel seste sonuç farklı:** Clotho'nun özgün testinde (1.045 kayıt / 5.225 sorgu) Hit@5, **EG2 ile %11,75, CLAP ile %37,42, birleşimleriyle %26,47**. CLAP farkı 25,67 yüzde puanı (%95 kaynak grubu güven aralığı: 23,25–27,96). Bu yüzden her ortamı ayrı ölçüyoruz. [Ses sonuçları ve eşleştirilmiş karşılaştırma](reports/multimodal/clotho-v2.1-evaluation-native-specialist/paired-comparisons.md). Ek alaka etiketleriyle yapılan protokol ayrı bir koşudur.

**BM25, altı kod ve sekiz belge koleksiyonunun tamamında 54.980 sorguyu bitirdi.** Yalnız pozitif kelime eşleşmeleri sonuçlara alındı; embedding, birleşim ve reranker deneyleri devam ediyor. [Başlangıç ölçümleri ve kesin sayılar](reports/multimodal/bm25-summary.md).

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
