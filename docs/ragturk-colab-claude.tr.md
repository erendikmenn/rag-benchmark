# RAGTurk: hazır vektörlerden Colab ve Claude CLI ile çalışma

16 Ekim 2026 tarihli kullanıcı talimatı yeni deney yürütmesini yetkilendirir. Bu protokol, önceki Gemma/MPS deneyinden ayrı kaydedilir. Eski deney kuyrukları ve kapatılmış otomasyon yeniden başlatılmaz.

## 22 Ekim: RAGTurk denetimi, metrikleri, yayın ve kesin CI tamamlandı

V11 üretimi ve son CPU denetimi **145.300 / 145.300 soru–koşul sonucunu** kapsıyor: **144.862 teknik olarak geçerli + 438 terminal geçersiz**, çözümsüz etkin istek sıfır. Geliştirme kapsamında **19.934 geçerli + 66 geçersiz = 20.000**, testte **124.928 geçerli + 372 geçersiz = 125.300** sonuç var. Üretici tamamlanma kaydının SHA-256 değeri `5f4e8e5b1910903a2282e6c33f761bb2a03f59607c38526944777671257adaac`; dosyanın 19 Ekim **03:59:46 İstanbul** zamanı dosya sistemi değişiklik zamanıdır, gömülü üretici zamanı değildir.

Tek nihai denetim 22 Ekim **16:02:28 İstanbul / 13:02:28 UTC** tarihinde başladı; **16:03:23 / 13:03:23** tarihinde tam bilimsel aşamaya girdi. Bilimsel aşama **17:41:46 / 14:41:46** tarihinde döndü; son kontroller ve tamamlanma kaydı **17:42:44.841079 / 14:42:44.841079** tarihinde bitti. Gerçek bekleme sonucu çıkış kodu **0**. Tek denetimin ölçülmüş toplam süresi **100 dakika 16,656 saniye**, bilimsel aşaması **98 dakika 23,336 saniye** oldu. Bu yalnız bu RAGTurk CPU koşusunun süresidir; gelecek süre tahmini veya diğer benchmarkların toplamı değildir. Ham yanıt, istek, kaynak, sıralama, sağlayıcı, token sınırı, bitiş ve metrik bağları denetlendi. **RAGTurk, yayımlanmış `a6e572cd` metrik sürümü için tamamlandı: dev/test metrikleri, bilimsel ve yayın bağımsız incelemeleri, yayın ve [kesin commit CI doğrulaması](https://github.com/erendikmenn/rag-benchmark/actions/runs/39801110850) geçti.** Bu tamamlanma yalnız RAGTurk'ü kapsar; diğer benchmark modaliteleri tamamlanmış sayılmaz.

Özgün süreçlerin kapanış kanıtını esas alan kalıcı düzeltme, diğer bütünlük kontrolleri korunarak kabul edildi. Denetim **yeni model çağrısı ve veritabanı yazımı yapmadı**; ücretli API veya yeni GPU tahsisi kullanılmadı. Kapalı cevaplar, 12 ayrı çözüm kaydı ve eski başarısız/belirsiz denemeler aynen korunur. İki tarihsel V8 denemesinin ve bir V10 tekrar denemesinin sağlayıcı tamamlanması hâlâ bilinmiyor. Özgün VM'nin ilk **3.360 soru** için son mühür eksikliği devam eder; sonraki gözlem özgün işçinin son mührü değildir. Kayıtsız, ayrılmış alt süreçlerin tamamının yokluğu iddia edilmez.

### Metriklerin kapsamı

Beş arama yöntemi × Laya kapalı/açık için her koşulun paydası **geliştirmede 2.000**, **testte 12.530** sorudur. Terminal geçersiz cevaplar kendi koşullarında **sıfır EM/F1** alır; ortak cevap kullanımı koşul paydalarını birleştirmez. Teknik geçerlilik cevap doğruluğu değildir. Tablolar **[0, 1]** ölçeğinde dört ondalığa yuvarlanmıştır. [Ayrı Claude sonuç JSON'u](../reports/ragturk-claude-full/summary.json) tam duyarlığı, 20 kanonik ve 17 ikincil arama metriğini, ayrıca yalnız geçerli cevaplara ait ikincil tanısal metrikleri içerir.

EM ve token F1, referansla **sözcüksel örtüşme** ölçer; anlamsal doğruluk, kaynağa sadakat veya atıf desteği ölçümü değildir. Bilinen sağlanmış kaynak atıfları mevcut puanlayıcıyla çıkarılır. İlk beş bağlam recall değeri, modele verilen beş pasajdaki altın kanıt kapsamıdır; soru hit oranı değildir. nDCG@10 tam aday sıralamasından hesaplanır. Arama metrikleri cevap geçerliliğinden bağımsız bütün soruları içerir. Altın etiketler yalnız CPU puanlayıcıda kullanıldı, cevap modeline verilmedi.

**Laya, iki bölümün tamamında beş yöntemin hepsinde F1'i, ilk beş bağlam kapsamını ve nDCG@10'u düşürdü.** Test F1 farkları **−6,03 ile −7,62 yüzde puan**, bağlam kapsamı farkları **−21,71 ile −28,56 puan** arasında. Geliştirme EM sonucu tek yönlü değildir: EmbeddingGemma 2'de +0,25 puan artış vardır. Eşleştirilmiş farklar betimseldir; **güven aralığı, p-değeri veya istatistiksel anlamlılık iddiası yoktur**. Bu sonuç her Laya/reranker kullanımının aynı davranacağı anlamına gelmez.

### Geliştirme: on koşul (her koşulda 2.000 soru)

| Arama | Laya | N | Geçersiz | Cevap EM | Token F1 | İlk beş bağlam recall | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| BM25 | Kapalı | 2000 | 5 | 0.0065 | 0.3627 | 0.8591 | 0.8434 |
| BM25 | Açık | 2000 | 4 | 0.0050 | 0.3054 | 0.6564 | 0.6033 |
| BGE-M3 | Kapalı | 2000 | 13 | 0.0080 | 0.3675 | 0.9008 | 0.8689 |
| BGE-M3 | Açık | 2000 | 5 | 0.0060 | 0.2972 | 0.6377 | 0.5925 |
| EmbeddingGemma 2 | Kapalı | 2000 | 11 | 0.0050 | 0.3615 | 0.8754 | 0.8403 |
| EmbeddingGemma 2 | Açık | 2000 | 7 | 0.0075 | 0.3001 | 0.6362 | 0.5904 |
| BM25 + BGE-M3 | Kapalı | 2000 | 4 | 0.0085 | 0.3721 | 0.9192 | 0.8909 |
| BM25 + BGE-M3 | Açık | 2000 | 5 | 0.0070 | 0.2988 | 0.6396 | 0.5917 |
| BM25 + EmbeddingGemma 2 | Kapalı | 2000 | 8 | 0.0090 | 0.3681 | 0.9089 | 0.8818 |
| BM25 + EmbeddingGemma 2 | Açık | 2000 | 4 | 0.0060 | 0.2986 | 0.6462 | 0.5948 |

### Geliştirme: beş eşleştirilmiş karşılaştırma (Laya − kapalı)

| Arama | Eşleşen N | Δ EM | Δ Token F1 | Δ İlk beş bağlam recall | Δ nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 2000 | -0.0015 | -0.0573 | -0.2027 | -0.2401 |
| BGE-M3 | 2000 | -0.0020 | -0.0703 | -0.2632 | -0.2764 |
| EmbeddingGemma 2 | 2000 | +0.0025 | -0.0614 | -0.2392 | -0.2499 |
| BM25 + BGE-M3 | 2000 | -0.0015 | -0.0733 | -0.2797 | -0.2993 |
| BM25 + EmbeddingGemma 2 | 2000 | -0.0030 | -0.0694 | -0.2628 | -0.2870 |

### Test: on koşul (her koşulda 12.530 soru)

| Arama | Laya | N | Geçersiz | Cevap EM | Token F1 | İlk beş bağlam recall | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| BM25 | Kapalı | 12530 | 40 | 0.0069 | 0.3614 | 0.8604 | 0.8378 |
| BM25 | Açık | 12530 | 22 | 0.0063 | 0.3011 | 0.6433 | 0.5912 |
| BGE-M3 | Kapalı | 12530 | 38 | 0.0058 | 0.3670 | 0.8979 | 0.8649 |
| BGE-M3 | Açık | 12530 | 43 | 0.0056 | 0.2977 | 0.6232 | 0.5745 |
| EmbeddingGemma 2 | Kapalı | 12530 | 56 | 0.0065 | 0.3642 | 0.8837 | 0.8449 |
| EmbeddingGemma 2 | Açık | 12530 | 32 | 0.0053 | 0.2980 | 0.6253 | 0.5788 |
| BM25 + BGE-M3 | Kapalı | 12530 | 46 | 0.0071 | 0.3725 | 0.9110 | 0.8876 |
| BM25 + BGE-M3 | Açık | 12530 | 31 | 0.0060 | 0.2963 | 0.6254 | 0.5743 |
| BM25 + EmbeddingGemma 2 | Kapalı | 12530 | 35 | 0.0062 | 0.3733 | 0.9081 | 0.8814 |
| BM25 + EmbeddingGemma 2 | Açık | 12530 | 29 | 0.0050 | 0.2971 | 0.6261 | 0.5771 |

### Test: beş eşleştirilmiş karşılaştırma (Laya − kapalı)

| Arama | Eşleşen N | Δ EM | Δ Token F1 | Δ İlk beş bağlam recall | Δ nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 12530 | -0.0006 | -0.0603 | -0.2171 | -0.2467 |
| BGE-M3 | 12530 | -0.0002 | -0.0694 | -0.2747 | -0.2904 |
| EmbeddingGemma 2 | 12530 | -0.0012 | -0.0662 | -0.2584 | -0.2661 |
| BM25 + BGE-M3 | 12530 | -0.0011 | -0.0762 | -0.2856 | -0.3133 |
| BM25 + EmbeddingGemma 2 | 12530 | -0.0012 | -0.0762 | -0.2819 | -0.3043 |

On koşulun tümü üzerinden ortalama EM/F1, geliştirmede **0,00685 / 0,333218**, testte **0,006089 / 0,332866** oldu. Bu ortalamalar matris özetidir; tek bir sistemin başarı olasılığı değildir. Laya kapalı BM25 + BGE-M3 geliştirmede en yüksek gözlenen F1'i (0,372130), Laya kapalı BM25 + EmbeddingGemma 2 testte en yüksek gözlenen F1'i (0,373310) aldı. Test sıralaması betimseldir; yeni ayarlama veya anlamlı fark iddiası taşımaz.

Bağımsız gerçek bilimsel denetim/metrik incelemesi SHA-256: `f88031eecb47ed3329de4b0b6282153537b42d4ead45424feef6bbc1c7696ff1`; root kabulü: `ca8a545d9ddec54bfe37d5266c23406a2fd1ebc24ee254c0132da9537a779656`. Bilimsel kabul 22 Ekim **17:52:40 İstanbul / 14:52:40 UTC** gözlemine bağlıdır. Yayın ve kesin CI kabulü **18:05:04 İstanbul / 15:05:04 UTC** gözleminde kaydedildi. Bağımsız yayın incelemesi SHA-256: `8d0cf0e85febcf15d1723103fe8b2a75aa288170b0939fea17e4b9d51dc087b2`. [Tamamlanma kaydı](../reports/ragturk-claude-full/completion.json), `a6e572cd5e41c69e9713ad9e8d033c96b4366812` metrik commit'i ile başarılı **39801110850** CI koşusunu bağlar; bu metadata güncellemesinin gelecekteki commit/CI sonucunu iddia etmez.

Denetim sonuç SHA-256: `6e284d5a73fee4dc83b8959e72979eb8442c371c596fceba04e7705760954856`. Tamamlanma kaydı: `67f682ecd7bbefa2c6190cdc65fe460a016df1820ed474dab28396b8ecaaa2f0`. Güvenli metrik özeti: `20304db4c048c5bf233d4c1feb316f0569d1af96e5fe5b6d33a467358875eba1`. [Yapılandırılmış ilerleme](../reports/ragturk-colab-claude-progress.json). Aşağıdaki üretim sürüyor veya denetim bekliyor ifadeleri **tarihli geçmiş gözlemleridir**; eski Gemma/MPS sonuçları ayrı deney olarak korunur.

## 17 Ekim: tarihsel süre uzatımı ve devam

17 Ekim 2026 saat **13:02 İstanbul** kapanış denetiminde Laya'nın bütün **14.530 soru / 909 blok / 72.650 liste / 3.632.412 gerçek aday çifti** bilgisayara kalıcı aktarıldı. Son 7 blok / 98 soru / 24.500 çift yeni byte, kaynak ve sıralama denetiminden geçti; önceki önek korundu, özgün 657 kimlik damgası değişmedi ve eski 210 blok yeniden taranmadı. Son alıcı, yerel istemci ve tam puanlama işi kapandı. Ayrı uzak sahiplik gözleminde gerçek işçinin bulunmadığı ve kernel'in boşta olduğu doğrulandı. Bu, **tam aktarım ve puanlama sürecinin kapanışıdır**; tamamlanma sonrası bağımsız byte ve aralık aşaması denetimi **13:09 İstanbul** gözleminde geçti. Denetim 174 kaynak/model/girdi dosyası ve yeni aralığın 699 arşivini kontrol etti; eski 210 arşivin tam taraması tekrarlanmadı. Bu aşamanın byte, receipt, sahiplik ve zaman bağları ayrı bağımsız incelemeden de geçti; bu inceleme tam benchmark denetimi değildir. Bu gözlem özgün işçinin ürettiği son mühür değildir; QA’nın tam ham yanıt/kaynak yeniden doğrulaması, metrik denetimi ve nihai bağımsız inceleme hâlâ bekliyor. Eski ilk 3.360 sorunun runtime son mührü eksikliği kalıcı olarak açıkça korunur.

QA'nın ayrı **13:07:44–13:07:46 İstanbul** salt okunur tek SQLite görüntüsünde **61.687 / 145.300 mantıksal koşul (%42,45) teknik olarak kapalıdır**: 61.452 geçerli + 235 teknik geçersiz. İki yeni çağrı çalışıyordu. QA o anda aktarılmış **909 blok / 14.530 Laya sorusunun tamamını** kaynak olarak kabul etmişti; bu, bunların bütün cevaplarının üretildiği veya doğrulandığı anlamına gelmez. Dev/test paydaları 20.000 / 125.300 olarak korunur. Bunlar cevap doğruluğu veya tamamlanmış metrik denetimi değildir.

Kullanıcının süre uzatma talimatıyla QA, genel süre kesicisi olmayan ayrı sürümde aynı ledger/istem/model, üç paralel CLI ve 2.048 token sınırıyla sürüyor. Devirdeki 20.866 kabul edilmiş kayıt, ilk 10.369 kayıt ve eski altı isteğin çözüm geçmişi korunmuştur. Önceki üreticinin kontrollü kapanışı ve yeni üreticinin ilk üç gerçek cevabı bağımsız doğrulandı; bu başlangıç denetimi sonraki bütün cevapların kalite denetimi değildir.

Uzak GPU başlatıcısının **30 saatlik kesicisi tarihsel olarak korundu; kaldırılmadı veya etkin kaynakta değiştirilmedi**. Gerçek puanlama işi bu süre dolmadan kapandı; artık bu tamamlanmış aralık için yeni GPU devri gerekmiyor. Yerel istemcinin bekleme sınırı tek başına uzak işin kapanış kanıtı sayılmadı. Aktarımın kapanış saati ile QA görüntüsünün saati ayrıdır.

Ek kullanım kapalıdır. Ücretli API, ek kredi ve yerel üretici model kullanılmaz; tamamlanmış geçerli veya teknik geçersiz cevaplar yeniden gönderilmez. İstek başına en çok üç CLI denemesi, 30/120 saniye bekleme ve çağrı süre sınırları değişmedi. Kota, giriş, ücretlendirme, bütünlük veya bilinmeyen hatalar kapanarak incelenir; belirsiz kayıtlar körlemesine sıfırlanmaz. Eski 3.360 soruluk VM aralığının son kaynak/model mührü eksikliği açık kalır. Tüm bloklar, 145.300 koşul, bağımsız dev/test metrik ve eşleştirilmiş karşılaştırma denetimi tamamlanmadan nihai benchmark sonucu yayımlanmaz. [Güncel güvenli durum](../reports/ragturk-colab-claude-progress.json).

## 16 Ekim 21:28 tarihli kurtarma gözlemi

16 Ekim 2026 saat 21:28:49 İstanbul gözleminde Claude QA devamı yeniden çalışıyor. Kullanıcının sonuçları koruyarak kurtarma ve devam talimatıyla, geçici sunucu/bağlantı hatalarını katı cevap kabulünden ayrı sınıflandıran sürüm devreye alındı. Geçerli veya çıktı sınırında teknik geçersiz kapanmış cevaplar tekrar gönderilmez. Yeni denemeler ayrı kayıtlarla tutulur; istek başına eski deneme dahil en çok üç CLI çağrısı, 30/120 saniye bekleme ve üç paralel çağrı sınırı vardır. Geçici hata sırasında sağlıklı paralel çağrıların kapanması beklenir. Kota, giriş, ek kullanım, model/kimlik veya bilinmeyen hatada otomatik devam yapılmaz.

**Önceki altı belirsiz isteğin tamamı yeni geçerli cevaplarla çözüldü.** Altı cevabın gerçek akış/istek/model/2.048 token/normal bitiş ve ek kullanımın devreye girmediğini bildiren kayıtları bağımsız doğrulandı. Eski 10.369 kabul edilmiş kayıt ile altı orijinal hata kaydı değişmedi; bu altı sonuç geçmişi silmeden eklenen ayrı çözüm kayıtlarıdır. Tek SQLite görüntüsünde **12.587 / 145.300 mantıksal koşul teknik olarak kapanmıştır**: 12.540 geçerli + 47 teknik geçersiz. O anda üç yeni çağrı çalışıyordu; bunlar eski altı sorunla karıştırılmaz. Altı sonucun bağımsız incelemesi, diğer yeni cevapların tamamının denetlendiği veya cevapların anlamsal olarak doğru olduğu iddiası değildir.

Colab'ın eski oturumunun artık tahsis listesinde bulunmadığı ayrıca doğrulandı. Mevcut kullanım hakkından tek yeni L4 kuruldu. Kalıcı **3.360 soru / 210 blok / 839.953 gerçek aday çifti**, kaynak ve sonuç bağlarıyla CPU üzerinde yeniden doğrulandı; GPU hesabı tekrarlanmadı. Kalan aralık 11.170 soru, 55.850 liste ve 2.792.459 gerçek aday çiftidir. 37 girdi parçasının tamamı gerçek byte/hash ile doğrulandı; önceden mevcut 15 parça tekrar yüklenmedi. CPU kabul kontrolünden sonra kalan aralık aynı bilimsel ayarlarla gerçek L4 üzerinde başladı. İlk yeni 50 adaylık listenin kaynak, istek, sıralama ve FP32 kullanım kayıtları CPU üzerinde doğrulandı; bu tek liste, tüm aralığın denetimi veya hız tahmini değildir. Ayrı 21:07:53 İstanbul aktarım denetiminde ilk iki yeni blok da bilgisayarda kapandı: 32 yeni soru / 160 liste / 8.000 çift. Kaynak, gerçek dosya hashleri, istek, sıralama ve oturum bağları doğrulandı. İlk iki blokla kalıcı kapsam 3.392 soruya ulaştı. Sonraki 21:29:45 İstanbul yerel denetiminde yalnız yeni 16 blok / 256 soru / 64.000 çift daha incelendi; toplam kalıcı kapsam **3.648 soru / 228 blok / 911.953 gerçek aday çifti** oldu. Önceki 657 dosyanın kimlik damgaları değişmedi; eski 210 blok yeniden taranmadı veya GPU üzerinde hesaplanmadı. Bir okuma zaman aşımı izinli sınırlı denemeyle aşıldı; alıcıda sonlandırıcı hata yoktu. Bu aktarım zamanı, yukarıdaki QA görüntüsünden ayrıdır. Tek sonuç alıcısı çalışıyor. Claude devamının ilk 500 isteklik grubunun tamamı, saat 21:29:44 İstanbul bağımsız denetiminde geçerli kapanışla doğrulandı: 3 eski belirsiz isteğin ayrı çözümü ve 497 yeni sonuç. Gerçek akış, istek, model, 2.048 token sınırı, normal bitiş, dosya ve SQL bağları eşleşti; 500 gözlenen kullanım olayı ek kullanımın devrede olmadığını bildirdi. Eski 10.369 kabul edilmiş kayıt ile altı orijinal kaydın dosyaları değişmedi. Bu denetim sonraki grupları veya cevapların anlamsal doğruluğunu kapsamaz.

Eski VM'nin son kaynak/model mührü kurtarılamadı. Eski aralık için başlangıç mühürleri, blok kimlikleri ve eldeki gerçek sonuçların denetimi korunur; kayıp son mühür varmış gibi raporlanmaz. Yeni aralık kendi son mührünü üretir. Eski VM'de kalıcı aktarılmamış son bölümün ne kadar hesaplandığı bilinmiyor. Tamamlanmış ve kalıcı doğrulanmış işler yeniden çalıştırılmıyor. Ücretli API, ek kredi veya yerel üretici model açılmadı. Özgün 48 saat sınırı uzatılmadı. Tam dev/test sonuçları, aralıkların kaynak/model geçmişi, 145.300 payda ve bağımsız metrik denetimi henüz tamamlanmadı.

## Tarihli önceki gözlemler

16 Ekim 2026 saat 19:40 İstanbul kontrolünde **QA ve checkpoint aktarımı durmuş durumdadır**. Yerel hata dosyaları 18:03 civarında kapanmıştır; bu zaman uzak GPU'nun durduğunu göstermez. QA'nın ikinci grubunda 398 geçerli sonuç ve saklanan kanıt hashleri korunmuştur. Sentetik sunucu hatası yanıtı katı akış denetiminde reddedilmiş; bir yeni fatal kayıt ve kesilen iki isteğin belirsiz kayıtları oluşmuştur. Önceki üç kayıt da değişmediğinden **toplam altı istek çözümlenmemiştir**; bunlar yeniden gönderilmedi veya sıfır puanlı geçersiz sonuca çevrilmedi. Kota tükenmesi, gerçek model değişimi veya ortak bağlantı nedeninin kesinliği kanıtlanmadı. Tek SQLite görüntüsünde **10.689 mantıksal koşul teknik olarak kapalıdır**: 10.646 geçerli + 43 teknik geçersiz; bu cevap doğruluğu veya nihai metrik değildir. Aktarım, oturum bilgisi okunurken `ConnectionError` ile kapanmıştır; bu hata sınırlı proof okuma tekrar yetkisinin dışında olduğundan tekrar yapılmamıştır. Ayrı 19:39:36 aktarım incelemesinde **3.360 soru / 210 blok / 839.953 gerçek aday çifti** bilgisayarda doğrulanmıştır: son iki blok ve 8.000 çift yeni denetlendi, önceki 148 kimlik damgası değişmedi, yarım dosya yoktur. Uzak GPU'nun güncel durumu bu yerel incelemede doğrulanmamıştır. Yeni devam başlatılmamış, kaynaklar ve başarısız kayıtlar korunmuştur. Aşağıdaki çalışıyor ifadeleri tarihli geçmiş gözlemlerdir.

16 Ekim 2026 saat 17:53 İstanbul gözleminde, yalnız hiç gönderilmemiş istekleri işleyen devamın **ilk 500 yeni isteği bağımsız denetimden geçti: 498 geçerli, 2 çıktı sınırında teknik geçersiz sonuç**. Gerçek yanıt dosyaları, istek/model/2.048 token sınırı ve ledger bağları doğrulandı. Bu 498 sayısı cevap doğruluğu değildir; iki teknik geçersiz sonuç ilgili EM/F1 paydalarında sıfır kalır ve tekrar edilmez. Eski bir fatal ve iki belirsiz istek ile eski başarısız grup değişmeden korunuyor. İkinci grup çalışıyor; denetim yalnız kapanmış ilk 500 isteği kapsar. Tek SQLite görüntüsünde toplam **10.595 mantıksal koşul teknik olarak kapanmıştır**: 10.552 geçerli + 43 geçersiz. QA, 2.000 dev ve ilk 1.200 test sorusunun kapalı Laya kaynaklarını kabul etmiştir; cevap ve metrik kapsamının tamamlandığı anlamına gelmez. Ayrı 17:53:16 aktarım denetiminde **3.328 soru / 208 blok / 831.953 gerçek aday çifti** kalıcı olarak doğrulandı; yalnız son 11 blok yeni hash denetiminden geçti, önceki 148 kimlik damgası değişmedi. Tam deney ve üç açık kaydın çözümü hâlâ bekliyor.

16 Ekim 2026 saat 17:32 İstanbul gözleminde yalnız hiç gönderilmemiş isteklere yönelik incelenmiş QA devamı çalışıyor. Aynı ledger, istem, model, üç paralel CLI ve 2.048 token sınırı korunuyor. 47 özel CPU kontrolü ve bağımsız inceleme geçti; daha önce kapanmış 9.471 sonuç salt okunur preflight ile yeniden doğrulandı. İlk üç yeni cevap da gerçek akış/istek/model bağları ve ek kullanımın devreye girmediğini bildiren kayıtlarıyla bağımsız kontrolden geçti. Son grupta seçilmiş 500 isteğin yalnız 279’unda gönderim izi vardı; hiç gönderilmemiş 221 istek tekrar çağrı sayılmadan sırada tutuldu. **Eski bir fatal ve iki belirsiz istek ile başarısız grubun kayıtları değişmedi; yeniden gönderilmedi ve sıfır puanlı geçersiz sonuca çevrilmedi.** Özgün 48 saat bütçe uzatılmadı; yeni hata otomatik devam oluşturmaz. Tek SQLite görüntüsünde **9.817 mantıksal koşul kapanmıştır**: 9.776 geçerli + 41 teknik geçersiz. Yeni grubun o anda 54 benzersiz geçerli sonucu vardı; bağımsız başlangıç denetimi yalnız ilk üçünü kapsar. Bu teknik ilerleme, cevap doğruluğu veya tam deney tamamlanması değildir. Colab ve aktarım sürüyor; ayrı 17:23:28 aktarım gözlemi 2.976 soru / 186 blok / 743.953 gerçek çifttir. Üç açık kayıt ile tam Laya ve metrik denetimleri tamamlanmadan nihai sonuç ilan edilmez.

Önceki 16 Ekim 2026 saat 17:07 İstanbul kontrolünde Claude QA sürecinin durduğu doğrulandı. CLI’ın sentetik sunucu hatası/API hatası yanıtı katı akış denetiminde reddedildi; kota tükenmesi, kota sıfırlama zamanı veya sunucu hatasının altında yatan neden kanıtlanmadı. Son gruptaki 276 geçerli cevap korunmuştur. Bir hata kaydı ile kesilmiş iki paralel isteğin belirsiz claim’leri değiştirilmemiş ve tekrar gönderilmemiştir. Tek SQLite görüntüsünde **9.760 mantıksal koşul kapanmıştır**: 9.719 geçerli + 41 teknik olarak geçersiz; bu üç çözümlenmemiş kayıt bunlara eklenmez veya sıfır puanlı geçersiz sonuç olarak yeniden etiketlenmez. Yalnız hiç gönderilmemiş istekler için ayrı devam sürümü hazırlanıyor; henüz çalıştırılmadı. Colab aktarımı sürüyor: ayrı 17:07:47 gözleminde **2.800 soru / 175 blok / 14.000 liste / 699.953 gerçek aday çifti** kalıcı olarak doğrulanmıştır. Üç açık kayıt, tam Laya son mühürleri ve bağımsız metrik denetimi çözülmeden tam tamamlanma ilan edilmez.

16 Ekim 2026 saat 16:07 İstanbul gözleminde 145.300 mantıksal sonucun **7.590’ı teknik olarak kapanmıştır**: 7.550 geçerli, 40 geçersiz koşul. Bu sayılar tek SQLite okuma işleminin aynı anlık görüntüsünden alınmıştır; doğruluk oranı değildir. QA kuyruğu 2.000 geliştirme sorusunun tamamına ve ilk 16 test sorusuna ait kapalı Laya kaynaklarını kabul etmiştir. Saat 16:08’de ayrı doğrulanan kalıcı aktarım kapsamı **2.096 soru / 131 blok / 10.480 liste / 523.953 gerçek aday çifti** olmuştur. Son aktarım kontrolü 11 yeni bloğu doğrulamış; önceki 148 dosyanın kimlik damgaları değişmemiştir. Kısa BM25 listeleri nedeniyle çift sayısı soru sayısının 250 katı olarak hesaplanmaz. Geliştirme cevapları, tam Laya kaynak/model son mühürleri ve bağımsız dev/test metrik denetimi henüz tamamlanmamıştır.

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

Aktarım süreci önce 208, ardından 368 soruluk kalıcı önekte okuma zaman aşımıyla yeniden denemeden durdu. İlk iki hata hangi HTTP aşamasında beklediğini kaydetmedi; nedenleri bilinmiyor. Tek seferlik salt okunur tanıda resmî oturum mevcut, kernel çalışır ve sonraki bloğun dosya bilgisi 0,71 saniyede erişilebilir bulundu. Bu gözlem önceki timeout nedenini kanıtlamaz. Ayrı sürüm hata aşamasını, türünü ve süreyi güvenli biçimde kaydeder; 8/30 saniyelik bağlantı/okuma ve 300 saniyelik toplam sınır değişmedi. 47 CPU kontrolü, gerçek yerel HTTP gecikme tekrarları ve bağımsız inceleme geçti. 23 eski bloğun 69 dosyası ve hata kanıtları korunarak tek bir incelenmiş devam başlatıldı. İlk yeni 368–384 bloğunun arşivi, özgün kanıtı ve aktarım makbuzu doğrulandı; bu gözlemde kalıcı kapsam 480 soru / 30 bloktur. GPU ve QA işi yeniden başlatılmadı; otomatik hata tekrarı yoktur.

Bu devam da 704 soru / 44 kapalı blokta durdu. Yeni hata kaydı, sonraki bloğun dosya bilgisi için `proof_contents` aşamasında 30,33 saniyelik `ReadTimeout` gösteriyor; önceki iki hatanın aşaması hâlâ bilinmiyor. 704 sorunun 3.520 listesi / 176.000 çifti Mac’te doğrulandı ve korundu; yarım dosya yok. Sonraki tek salt okunur kontrolde aynı oturumun kernel’i çalışır durumdaydı ve 704–720 bloğunun dosya bilgisi 1,47 saniyede geldi. Bu ölçüm kalıcı erişim sorunu veya timeout’un altında yatan nedeni kanıtlamaz. Bu hata sonrasında aktarım, yeni devam için kullanıcı onayı gelene kadar kapalı kaldı. Claude QA kuyruğu mevcut kaynaklarla çalışmayı sürdürüyor.

Ayrı devam önerisi 44 CPU testi ve bağımsız incelemeden geçti. Öneri, yalnız dosya bilgisi okumasındaki `ReadTimeout` için toplam en fazla üç GET denemesi ve aralarda 2/5 saniye bekleme öngörür; 8/30 ve 300 saniyelik sınırlar korunur. Her denemeden önce niyet, ardından sonuç değiştirilemeyen özel kayda yazılır. Diğer hata türleri, arşiv aktarımı ve model çağrıları tekrar edilmez. 44 eski bloğun 132 dosyası ve toplam 148 mühür devam planına bağlandı. Kullanıcı bu sınırlı devamı doğrudan onayladı; onay kaydı tam plan kimliğine bağlanarak tek aktarım süreci başlatıldı. İlk 704–720 bloğunun arşivi, özgün kanıtı, aktarım makbuzu ve deneme kayıtları bağımsız doğrulandı. İlk dosya bilgisi isteği tek denemede HTTP 200 aldı. 16 Ekim 11:48:31 UTC gözleminde kalıcı kapsam 848 soru / 53 blok / 4.240 liste / 212.000 çifttir. Önceki 148 dosya mührü değişmedi; GPU ve QA işleri yeniden başlatılmadı. Bu kısmi aktarım doğrulaması, tam Laya skor denetimi veya benchmark tamamlanması değildir.

LLM tam kimliği **`claude-haiku-5-5`**, taşıma **Claude Code CLI 2.1.291 / abonelik oturumu**. Çocuk süreç ortamından API anahtarı ve sağlayıcı/model yönlendirmeleri çıkarılır; ilk taraf abonelik girişi her çağrı öncesi kontrol edilir. Araçlar/MCP kapalı, taze oturum, yalnız soru ve kaynak pasajlar kullanılır. Tanı protokolünün çıktı sınırı 512, yeni V3 protokolünün açık sınırı 2.048 tokendir; reasoning `adaptive/low` olarak kaydedilir. CLI temperature/seed ayarı sunmadığı için bunlar sıfırmış gibi raporlanmaz. Kesik veya model kimliği uyuşmayan cevap kabul edilmez.

Kullanıcı ek kullanımın kapalı olduğunu doğruladı. İlk gerçek teknik yanıtın kullanım kaydı da `isUsingOverage=false` gösterdi. CLI'ın dolar tahmini fatura değildir. Birden çok CLI ortak hesap limitini kullanır; paralellik kota oluşturmaz. Colab hakkı tükenirse uyumlu arama/reranker işleri Mac'te ayrı runtime kimliğiyle sürebilir. Claude abonelik kotası dolarsa cevap işi reseti bekler; ücretli API veya yerel üretici LLM'e otomatik geçiş yoktur.

Tam veri koşusundaki ilk kapalı geçersiz cevap, 2.048 token sınırında `max_tokens` ile durduğu için bağımsız doğrulandı. Bu kayıt yeniden gönderilmez ve cevap EM/F1 hesabına sıfır katkıyla dahil edilir. Kaynak bütünlüğü veya faturalama koruması hatası değildir; 20 soruluk pilotun tüm cevaplarının geçerli olması, tam veride kesilme olmayacağını garanti etmez.

Projeksiyon düzeltmesinden sonraki ilk 500 yeni isteğin kapalı denetimi 498 geçerli ve 2 `max_tokens` nedeniyle geçersiz sonuç doğruladı. Model, 2.048 token sınırı, kaynak ve devam sürümü bağları eşleşti; önceki 500 cevabın mühürleri değişmedi. Bunlar teknik sonuç sayılarıdır; tam veri kalite değerlendirmesi bekliyor.

Her gönderim yeni özel claim ve kapalı çıktı kanıtıyla kaydedilir. Geçerli kapanmamış bir istekte tekrar, güncel sınırlı kurtarma politikasının sınıflandırılmış kapalı kanıtına bağlıdır; belirsiz yeni çağrı doğrudan tekrarlanmaz. Tamamlanan veri aynı istem kimliğiyle doğrulanmadan cache olarak kabul edilmez. Ham soru, kaynak metin, yanıt ve kimlik bilgileri GitHub'a konmaz. Yayımlanan sayılar, yöntemler, süreler ve hash kanıtları içerik taşımaz.

## Ölçüm sınırları

20 soruluk pilot, model kalitesinin tam veri sonucu değildir. Geçerli cevap sayısı, normal bitiş, doğru kaynak bağlama ve kapsama kontrolünden sonra cevap EM/token-F1 ve retrieval metrikleri hesaplanır. İstem paylaşımı yapılan yanıtların latency'si yeni çağrı latency'si olarak sayılmaz. Colab tahsisi ve Claude kotası değişken olduğu için tam bitiş saati henüz ölçülmüş değildir.

Claude'un [abonelikle programatik kullanım](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan), [headless CLI](https://code.claude.com/docs/en/headless) ve [ek kullanım](https://support.claude.com/en/articles/12429409-manage-usage-credits-for-paid-claude-plans) belgeleri bu taşımanın hesabın ortak kullanım sınırlarına tabi olduğunu açıklar.

Tam Laya işinde 3.632.412 gerçek aday çifti vardır. Pilotun model yüklemesi dışındaki 4.950 çift / 98,11 saniye ölçümü tek L4 üzerinde yaklaşık 20 GPU-saat puanlama senaryosu verir. Bu, veri aktarımı/denetim, oturum ömrü, tam verinin uzunluk dağılımı veya Claude kotası dahil bitiş tahmini değildir.
