# rag-benchmark

**Türkçe RAG sistemlerini aynı koşullarda, yerel modellerle karşılaştırmak için bir deney düzeneği.** BM25, BGE-M3 ve EmbeddingGemma 2 belge aramasını; Laya'nın yeniden sıralamaya katkısını; aynı Gemma 4 modelinin ürettiği cevapları ölçer.

[English](README.md) · [Deney protokolü](docs/protocol.md) · [Sonuçlar ve doğrulama durumu](docs/results.md)

**Mevcut durum:** indirilebilir veri denetlendi: **37.511 metin parçası ve 14.530 kullanılabilir soru–cevap**; 2.000 geliştirme, 12.530 test sorusu. Model bağlantıları doğrulanıyor. On varyantın tamamını kapsayan gerçek model sonucu henüz yayımlanmadı. Küçük bir teknik kontrol, tam benchmark sonucu sayılmıyor.

## Sistem ne yapıyor?

RAG, cevap verecek modele soruyla ilgili kaynak metinleri önceden sunmaktır. Örneğin kullanıcı “Bu kurum ne zaman kuruldu?” diye sorar:

1. **Arama:** sistem bütün kaynaklar arasından ilgili olabilecek en fazla 50 metin getirir.
2. **Yeniden sıralama:** Laya açıksa bu adayları soru açısından değerlendirir; en yüksek puanlı en fazla 5 metin seçilir. Kapalıysa aramanın ilk 5 sonucu kullanılır.
3. **Cevap:** aynı yerel Gemma 4 modeli soruyu ve bu 5 metni alıp kaynak kimlikleriyle cevap yazar.

BM25 kelime örtüşmesini, embedding modeli anlam yakınlığını ölçer. Hibrit kolda iki arama paralel yapılır, sonuç sıraları birleştirilir. **BGE-M3 bu deneyde embedding modelidir; ayrıca kullanılan bir BGE reranker yoktur.**

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
```

`serve`, GGUF dosyasının SHA-256 özetini doğrular; llama.cpp'yi Metal ile `http://127.0.0.1:8080/v1` adresinde, `gemma4` adıyla çalıştırır. Bu yerel protokol, OpenAI hizmeti veya API anahtarı kullanmaz. GGUF dosyası ve doğrulama özeti [`configs/ragturk.toml`](configs/ragturk.toml) içindedir. Cevap üretim ayarları tüm kollarda sıcaklık 0, seed 42, en fazla 256 çıktı tokenı ve düşünme modu kapalıdır. Teknik ayrıntılar [model sözleşmeleri](docs/models.md) sayfasında.

Yalnız aramayı sınamak için `run` komutuna `--retrieval-only` eklenebilir; bu mod cevap üretmediğinden uçtan uca RAG sonucu değildir. Hazırlanan veriler `data/ragturk/` altında tutulur. Yöntem kimlikleri `bm25`, `bge`, `embeddinggemma`, `bm25_bge`, `bm25_embeddinggemma`; Laya'lı kola `_laya` eklenir.

Önce 20 soruluk geliştirme kontrolü yapılır. Tam deney bundan sonra çalıştırılır. İlk veri/model indirmeleri internet gerektirir; gerekli dosyalar hazır olduğunda değerlendirme yerel çalışacak şekilde tasarlanır. Ücretli bir inference API'si kullanmak planın parçası değildir. Yerelde çalışmak donanım, bellek ve elektrik maliyetini ortadan kaldırmaz.

## Sonuçları nasıl okumalı?

- **Recall@50:** ilgili diye etiketlenmiş kaynakların ne kadarı ilk 50 adaya girdi? Buraya girmeyen kaynağı Laya geri getiremez.
- **Recall@5:** ilgili kaynakların ne kadarı son seçilen en fazla 5 metin içinde?
- **MRR/nDCG:** doğru kaynak sıralamanın ne kadar başında?
- **Cevap EM/F1:** üretilen cevap, referans cevabın kelimeleriyle ne kadar örtüşüyor?
- **Gecikme:** arama, Laya ve cevap üretimi ayrı ayrı ne kadar sürüyor?

**%99 kaynak bulma, %99 doğru cevap anlamına gelmez.** Model doğru kaynağı görüp yanlış cevap verebilir. Doğru bir cevabın farklı kelimelerle yazılması da F1'i düşürebilir; bu metrik tek başına insanın yaptığı olgusal doğruluk denetimi değildir.

Laya'nın verdiği 0–1 puanı bu görev için doğrulanmış bir güven yüzdesi değildir. İlk deneyde düşük puanlı her metni otomatik silen eşik kullanılmaz; puana göre ilk 5 seçilir. Eşik, prompt veya model ayarı gerekiyorsa geliştirme verisinde seçilir; son test sonuçlarına bakılarak ayar değiştirilmez.

Önbellekteki aynı cevabı yeniden kullanmak zaman kazandırabilir. Ancak önbellekten okuma süresi, modelin yeni cevap üretme hızına dahil edilmez. Deneyde soru embedding'leri yeniden hesaplanır; hazırlık sonrası ilk sorunun süresi de yüklenmiş model gecikme özetinden çıkarılır. İlk yükleme, gerçek hesaplama ve önbellek isabeti ayrı koşullardır.

## Lisans

Projenin kodu **MIT** lisanslıdır. RAGTurk veri yayını **CC BY-NC-SA 4.0** olarak etiketlenmiştir; model ağırlıkları da kendi koşullarını korur. Kodun açık kaynak olması, veri ve modelleri otomatik olarak sınırsız ticari kullanıma açmaz. Ham veri ve model ağırlıkları bu depoda yayımlanmaz.

Bu bağımsız bir değerlendirme projesidir. Makaledeki veya önceki Jev benchmark'ındaki sonuçlar, bu yeni deney yapılmış gibi gösterilmez. Ayrıntılar için [protokol](docs/protocol.md) ve [sonuç durumu](docs/results.md).
