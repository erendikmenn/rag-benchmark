# RAGTurk: hazır vektörlerden Colab ve Claude CLI ile çalışma

16 Ekim 2026 tarihli kullanıcı talimatı yeni deney yürütmesini yetkilendirir. Bu protokol, önceki Gemma/MPS deneyinden ayrı kaydedilir. Eski deney kuyrukları ve kapatılmış otomasyon yeniden başlatılmaz.

## Kaynaklar ve ortak hesaplama

37.511 chunk ve 14.530 sorunun BGE-M3 / EmbeddingGemma 2 vektörleri ile özgün BM25 indeksi hazırdır. Kaynak, model, sıralama ve dosya kimlikleri eşleşen bu nesneler tekrar kullanılır. Yeni embedding üretilmez. Geliştirme (2.000) ve test (12.530) sonuçları ayrı tutulur; etiketler cevap üreten modele gönderilmez.

Hazır FP32 matrislerde CPU batch araması ve BM25 ile bütün 14.530 soru için beş yöntemden **72.650 aday listesi** üretildi. Kaynaklar, 114 çıktı bloğu, sıra, galeri üyeliği ve RRF birleşimleri bağımsız doğrulandı. BM25'de dört liste 50'den kısa; hiçbiri boş değil. Liste uzatmak için yapay aday eklenmez. Bu kayıtlar henüz cevap doğruluğu ölçümü değildir. Batch matris çarpımı, tarihsel tek-sorgu hesaplamasıyla byte eşitliği iddiası taşımaz.

## On koşul

| Koşul | Arama | Yeniden sıralama | Cevap modeli |
|---|---|---|---|
| 1 | BM25 | Yok | Claude Haiku 5.5 |
| 2 | BM25 | Laya | Aynı model ve istem |
| 3 | BGE-M3 dense | Yok | Aynı model ve istem |
| 4 | BGE-M3 dense | Laya | Aynı model ve istem |
| 5 | EmbeddingGemma 2 dense | Yok | Aynı model ve istem |
| 6 | EmbeddingGemma 2 dense | Laya | Aynı model ve istem |
| 7 | BM25 + BGE, RRF | Yok | Aynı model ve istem |
| 8 | BM25 + BGE, RRF | Laya | Aynı model ve istem |
| 9 | BM25 + EmbeddingGemma 2, RRF | Yok | Aynı model ve istem |
| 10 | BM25 + EmbeddingGemma 2, RRF | Laya | Aynı model ve istem |

Her kanalın ilk 100 adayı bir kez hesaplanır; RRF sabiti 60'tır. Her koşulun en çok 50 adayı vardır; cevap aşamasına ilk beş pasaj girer. Aynı kaynak listesi none/Laya kollarının ortak başlangıcıdır. Laya bu listeyi yeniden sıralar. Farklı retrieval yöntemlerinin embedding kimlikleri birleştirilmez. Tam istem ve model/adapter kimliği aynıysa bir tamamlanmış cevap birden fazla koşulda kullanılabilir; her koşulun etiketi ve değerlendirmesi ayrı kalır.

## Yürütme ve paralellik

Mevcut Colab hakkından bir NVIDIA L4 tahsisi doğrulandı. Laya'nın sabit revision'ı `1720e3e3357cfe1e281542e223f8273b0890ca34`; SDK/eager, CUDA, FP32, batch 8, max_len 8192, eşik kapalıdır. Bu yeni CUDA koşulu tarihsel MPS sonucu olarak etiketlenmez. Aynı GPU'da bir model sahibi, aday listelerinin kuyruğunu işler. Bağımsız LLM istekleri Mac'teki CLI üzerinden uzak Claude hizmetinde paralel ilerler. Mantıksal on koşul, on GPU veya on model kopyası anlamına gelmez.

İlk kontrol 20 geliştirme sorusudur: 100 aday listesi ve none/Laya birlikte 200 mantıksal soru-koşul satırı. Bu, tam veri kapsamının yerine geçmez. Üç paralel CLI isteğinin ikisi kabul edildi; üçüncü istekte CLI kendi çıktı-sınırı devam mesajını ekledi, iki tur/842 toplam token üretti ve katı kontrol bunu reddetti. `--max-turns 1` ve HTTP retry kapatma bu iç devam yolunu engellemedi. Yeni akış denetimiyle 20 sorunun 200 mantıksal koşulu tamamlandı: 192 farklı istemin 133'ü geçerli, 59'u çıktı sınırında geçersiz; ortak istemlerin koşullara dağılımıyla 140 geçerli / 60 geçersiz satır vardır. Geçersiz satırlar EM/F1 paydasında sıfır olarak tutulur. Bu %30 sınır etkisi nedeniyle 512 token protokolü tam koşuya taşınmadı. Aynı model ve istemle ayrı 2.048 token protokolünün 20 soruluk pilotu tamamlandı: 192 benzersiz isteğin ve 200 mantıksal koşulun tamamı geçerli normal bitişle kapandı. Bağımsız denetim kaynak/istem bağlarını, tüm kapalı yanıtları, 2.048 sınırını ve ek kullanımın kapalı olduğunu bildiren 192 kullanım olayını doğruladı. Tam veri dağıtıcısının bağımsız incelemesi ve CI geçti; ilk üç yeni isteğin kapanması doğrulandıktan sonra tam QA koşusu üç paralel CLI isteğiyle başladı. Kuyruk en çok 500 yeni isteklik gruplar halinde ilerler; toplam kapsam 14.530 × 10 = 145.300 mantıksal sonuçtur. 512 token sonuçları değiştirilmez veya yeni protokolle birleştirilmez. Laya'nın 100 liste/5.000 çift kaydı ise SDK tokenizer dönüşümünün birebir CPU yeniden üretimi, diğer 49 kaynak/model mührü ve tüm kayıt bağlarının bağımsız denetimiyle ayrı bir kabul kanıtı aldı; özgün başarısız kayıt korunur. Yeni GPU çıkarımı yapılmadı. [Güncel güvenli durum](../reports/ragturk-colab-claude-progress.json).

512 token pilotunun [bağımsız metrik raporu](../reports/ragturk-claude-pilot-v2-metrics.json), 20 geliştirme sorusuna aittir; tam veri sonucu veya semantik doğruluk ölçümü değildir. Her koşulun geçersiz satır sayısı ayrı gösterilir. İlk tam koşulara geçerken geliştirme/test ayrımı ve protokol sabit tutulur. Genel amaçlı bütün modaliteleri çalıştıran yeni dağıtık scheduler henüz tamamlanmış değildir.

2.048 token pilotunun [ayrı bağımsız metrik raporunda](../reports/ragturk-claude-pilot-v3-metrics.json) bütün yanıtlar teknik olarak geçerlidir. Bu küçük örnekte Laya, beş arama yönteminin tamamında ilk beş kaynak kapsamasını ve cevap token-F1 değerini düşürdü. Bu gözlem korunur; model lehine sonuç çıkarmak için protokol değiştirilmez. Ortalama cevap token-F1 0,3256 ve tam metin eşleşmesi 0’dır; bunlar referansla sözcük örtüşmesini ölçer, insan değerlendirmesi veya anlamsal doğruluk oranı değildir.

Önceki pilot L4 oturumu resmî tahsis listesinden kayboldu; nedeni bilinmiyor. Pilot kanıtları Mac’te doğrulanmış yedektedir. Mevcut kullanım hakkından tek bir yeni L4 oturumu ayrıldı; gerçek runtime ve CPU kabul denetiminden sonra tam Laya işi başladı. İlk 32 soru / 160 liste / 8.000 çift, iki kapalı blok olarak Mac’te hash ile doğrulandı. Kapanmış sonuç blokları bilgisayara aktarılmadan yalnız VM’deki dosya kalıcı yedek sayılmaz. Aktarım makbuzu en son atomik olarak yayımlanır; QA kuyruğu ancak kaynak/rank/claim bağlarını denetlediği kapalı blokları kullanır. Tamamlanma için ayrıca tüm Laya işinin kaynak/model son mühürleri ve bağımsız tam veri denetimi gerekir.

İlk kapalı Laya bloklarının bağımsız denetimi, yerel kabul kodunda metadata projeksiyonu hatası buldu: uzak iş yalnız `id/title/text` alanlarını hash’lerken yerel doğrulayıcı ek `article_id/source` alanlarını da katıyordu. İlk 32 sorunun gerçek kaynakları, skorları ve sıralamaları doğru projeksiyonla doğrulandı. İlk 500 yeni Claude isteği geçerli kapanışla tamamlandı ve bağımsız doğrulandı. Eski supervisor, Laya kabulünde ledger’a yazmadan durdu. Ayrı V2 devam sürümü yalnız bu kaynak alanı eşleştirmesini düzeltti; 57 CPU testi, bağımsız kaynak incelemesi ve gerçek iki arşivin 160 listesinin yeniden doğrulanması geçti. Aynı ledger, temel plan, istem, model ve 2.048 token sınırı korunarak devam başlatıldı; kapalı Laya blokları artık kabul ediliyor. Eski başarılı cevaplar, claim’ler ve kaynak mühürleri değiştirilmedi; Laya hesaplaması tekrarlanmadı.

LLM tam kimliği **`claude-haiku-5-5`**, taşıma **Claude Code CLI 2.1.291 / abonelik oturumu**. Çocuk süreç ortamından API anahtarı ve sağlayıcı/model yönlendirmeleri çıkarılır; ilk taraf abonelik girişi her çağrı öncesi kontrol edilir. Araçlar/MCP kapalı, taze oturum, yalnız soru ve kaynak pasajlar kullanılır. Tanı protokolünün çıktı sınırı 512, yeni V3 protokolünün açık sınırı 2.048 tokendir; reasoning `adaptive/low` olarak kaydedilir. CLI temperature/seed ayarı sunmadığı için bunlar sıfırmış gibi raporlanmaz. Kesik veya model kimliği uyuşmayan cevap kabul edilmez.

Kullanıcı ek kullanımın kapalı olduğunu doğruladı. İlk gerçek teknik yanıtın kullanım kaydı da `isUsingOverage=false` gösterdi. CLI'ın dolar tahmini fatura değildir. Birden çok CLI ortak hesap limitini kullanır; paralellik kota oluşturmaz. Colab hakkı tükenirse uyumlu arama/reranker işleri Mac'te ayrı runtime kimliğiyle sürebilir. Claude abonelik kotası dolarsa cevap işi reseti bekler; ücretli API veya yerel üretici LLM'e otomatik geçiş yoktur.

Tam veri koşusundaki ilk kapalı geçersiz cevap, 2.048 token sınırında `max_tokens` ile durduğu için bağımsız doğrulandı. Bu kayıt yeniden gönderilmez ve cevap EM/F1 hesabına sıfır katkıyla dahil edilir. Kaynak bütünlüğü veya faturalama koruması hatası değildir; 20 soruluk pilotun tüm cevaplarının geçerli olması, tam veride kesilme olmayacağını garanti etmez.

Her gönderim yeni özel claim ve kapalı çıktı kanıtıyla kaydedilir. Başarısız/belirsiz istek otomatik tekrarlanmaz. Tamamlanan veri aynı istem kimliğiyle doğrulanmadan cache olarak kabul edilmez. Ham soru, kaynak metin, yanıt ve kimlik bilgileri GitHub'a konmaz. Yayımlanan sayılar, yöntemler, süreler ve hash kanıtları içerik taşımaz.

## Ölçüm sınırları

20 soruluk pilot, model kalitesinin tam veri sonucu değildir. Geçerli cevap sayısı, normal bitiş, doğru kaynak bağlama ve kapsama kontrolünden sonra cevap EM/token-F1 ve retrieval metrikleri hesaplanır. İstem paylaşımı yapılan yanıtların latency'si yeni çağrı latency'si olarak sayılmaz. Colab tahsisi ve Claude kotası değişken olduğu için tam bitiş saati henüz ölçülmüş değildir.

Claude'un [abonelikle programatik kullanım](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan), [headless CLI](https://code.claude.com/docs/en/headless) ve [ek kullanım](https://support.claude.com/en/articles/12429409-manage-usage-credits-for-paid-claude-plans) belgeleri bu taşımanın hesabın ortak kullanım sınırlarına tabi olduğunu açıklar.

Tam Laya işinde 3.632.412 gerçek aday çifti vardır. Pilotun model yüklemesi dışındaki 4.950 çift / 98,11 saniye ölçümü tek L4 üzerinde yaklaşık 20 GPU-saat puanlama senaryosu verir. Bu, veri aktarımı/denetim, oturum ömrü, tam verinin uzunluk dağılımı veya Claude kotası dahil bitiş tahmini değildir.
