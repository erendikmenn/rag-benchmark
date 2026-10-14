# Google Colab ile paralel devam

14 Ekim 2026: **üç gerçek GPU benchmark işi paralel çalışıyor.** Fizik FR, A100-SXM4-80GB; Pharmaceuticals EN ve Industrial EN ayrı NVIDIA L4 çalışma zamanlarında. Üçünde de kaynak `98a981ed292b49f519ec2e6555cc1ab0cd929976`, doğal CUDA BF16 ve aynı deney koşulları korunur. Yerel Mac kuyruğu yeniden başlatılmadı.

| Koleksiyon | Doğrulanmış hazırlık | Tamamlanan kanallar | Çalışan kanal | Gözlenen kanal duvar süresi |
|---|---:|---|---|---|
| Physics FR | 215,157 sn | B/G/E/N | S | G 36,260; E 38,298; N 255,382 sn |
| Pharmaceuticals EN | 160,396 sn | B/G | E | G 50,470 sn |
| Industrial EN | 206,378 sn | B | G | Henüz kayıt yok |

Kanal duvar süreleri **checkpoint oluşturma dahil** gözlemlerdir; yalnız model çıkarım süresi veya GPU hızlanma katsayısı değildir. Henüz tam 126 hücrelik koşu veya bağımsız sonuç denetimi bitmedi. Yayımlanmış ana kapsam **2.052 / 18.460** olarak kalır. [Teknik operasyon özeti](../reports/colab-runtime-readiness.json).

Büyük yerel medya arşivi taşıma darboğazı, toplam **5.213.335 baytlık üç küçük metadata arşivi** ve Colab'dan doğrudan sabit resmî corpus dosyalarını alma yöntemiyle giderildi. Arşivler özgün altı metadata dosyasının birebir baytlarını korur; her corpus shard'ının ve gömülü medya dosyasının SHA-256'sı doğrulandı. Üç özgün veri kimliği ve worker dry-run geçti; metadata yeniden üretilmedi. Bir değişmez Physics checkpoint arşivi (53.514.240 bayt) Mac’e indirildi ve SHA-256 doğrulandı. Bu kısmi koşu yedeğidir; tamamlanmış benchmark değildir.

**Tarihsel iki runtime kaybı:** ilk L4, ortam/model dosyası hazırlığından sonra veri aktarılırken kayboldu. Daha sonraki A100 girişimi de 146 parçanın 120'si aktarılmışken erişilemez oldu. İkisinin nedeni bilinmiyor; bu eski başarısız girişimlerde sinirsel benchmark başlamamıştı. Bu tarihsel durumlar yukarıdaki üç mevcut çalışma zamanının durumu değildir; özgün yerel paketler korunuyor.

Ölçüme dayanmayan önceki 11 günlük tamamlanma tahmini geri çekildi. Görev sayısı benzersiz model çıkarımı sayısı değildir. İlk üç arama koleksiyonu, kalan ana matris ve QA ayrı kapsamlar olarak ölçülecek; bir saatlik veya gece boyunca tamamlanma sözü verilmiyor. Kısmi kanal süreleri toplam araştırma için sayısal ETA sağlamaz.

[Çevrimdışı QA istek sayımı](../reports/colab-qa-request-count-readiness.json), 540.842 uygun görevi aynı doğrulanmış runtime koşulunda 122.110 birebir istek anahtarına gruplar. 20.020 metin sınırı, 30 kullanılamaz metin ve 1.272 eski rapor görevi ayrı tutulur; model çağrısı 0, gerçek cache kazancı ve süre ölçülmedi.

## İlk dalga

Her notebook ayrı bir Colab çalışma zamanında bir tam veri kümesi işler. İlk dalga, yerelde tam dense araması yapılmamış üç koleksiyonu kullanır. Bu güncellemede üç eşzamanlı GPU çalışma zamanı doğrulandı; gelecekteki tahsis veya oturum süresi garanti edilmez.

| Notebook | Tam sorgu | Sayfa | Hedef |
|---|---:|---:|---|
| [Fizik FR](https://colab.research.google.com/github/erendikmenn/rag-benchmark/blob/main/notebooks/colab/vidore-v3-physics-fr.ipynb) | 302 | 1.674 | 126 none koşulu |
| [İlaç EN](https://colab.research.google.com/github/erendikmenn/rag-benchmark/blob/main/notebooks/colab/vidore-v3-pharmaceuticals-en.ipynb) | 364 | 2.313 | 126 none koşulu |
| [Endüstri EN](https://colab.research.google.com/github/erendikmenn/rag-benchmark/blob/main/notebooks/colab/vidore-v3-industrial-en.ipynb) | 283 | 5.244 | 126 none koşulu |

Toplam 949 sorgu, 9.231 sayfa ve 378 hedef koşul. Bunlar plan sayılarıdır. Her koşul tam galeri ve tam sorgu kümesini kullanır; rastgele sorgu bölme yoktur. B/G/E/N/S/J kanalları ve iki aday bütçesi korunur. Reranker, QA, Gemma açıklaması ve BGE token-head deneyleri bu ilk dalgada yoktur. Bunların mevcut teknik engelleri ayrı ele alınır.

CUDA BF16 sonuçları yeni koşullardır. Mac MPS cache/SQLite dosyaları CUDA kimliğiyle yeniden etiketlenmez. Ham verinin taşınması kaynak kimliğini korur; donanım/paket değişikliği sayısal sonuçları değiştirebilir. Tamamlanan CUDA sonuçları bağımsız denetim ve kapsam kararı verilene kadar ana matrise eklenmez. Önceki CS/FLEURS özel analizlerine dokunulmaz.

## Notebook kullanımı

1. Notebooku açıp Drive'a kendi kopyanızı kaydedin. GPU çalışma zamanı seçin. Bu plan doğal BF16 ister; L4/A100 gibi uygun bir cihaz gerekir. T4/CPU'ya sessiz geçiş yapılmaz. Gerçek GPU, CUDA sürümü, paketler ve kaynak hash'leri kaydedilir.
2. Yerelde `work/colab/bundles/` altında hazırlanan ilgili özel `.tar` dosyasını Drive'daki `MyDrive/rag-benchmark-colab/input/` klasörüne koyun. Üç arşiv toplam yaklaşık 4,07 GB'dır. Bunlar GitHub'a yüklenmez; kamuya açık checksum listesi [buradadır](../configs/colab-bundle-checksums.json).
3. Kurulum hücresi kaynak kodu tam commit SHA'sına sabitler ve `uv.lock` ile ayrı Python 3.12 ortamı kurar. Aktif Mac ortamı değişmez. Model dosyaları sabit Hugging Face revision'larından ayrıca indirilir. Model erişimi gerekiyorsa ilgili koşulları hesabınızda kabul edip Colab Secrets'a `HF_TOKEN` ekleyin; tokenı hücreye veya sohbete yazmayın.
4. Veri doğrulama ve plan hücrelerini çalıştırın. `RUN_FULL=False` varsayılanı gerçek deneyi başlatmaz. Model dosyası indirme hücresi GPU çıkarımı yapmaz.
5. Tam koşu için `RUN_FULL=True` yapın. Aynı işe açık devam gerektiğinde `RESUME=True` seçilir. Üç ayrı notebook yalnız üç ayrı çalışma zamanında paralel çalıştırılır. Aynı işi iki oturuma vermeyin; merkezi atamayı bu sohbetten yöneteceğiz.

Her ağır kanal ayrı alt süreçte çalışır; bir kanal bittiğinde süreç kapanır. Altı kanal tamamlandıktan sonra birleşimler aynı CUDA cache'inden hesaplanır. Bu, önceki yerel bellek hatalarının kök nedeninin çözüldüğü veya GPU başarısının ölçüldüğü anlamına gelmez.

Her VM'de ortak bir dosya kilidi ikinci worker'ı engeller. SQLite ve çıkarım cache'i `/content/rag-work/` üzerinde tutulur; canlı veritabanı Drive üzerinde çalıştırılmaz. Her tamamlanan kanal ve son aşamadan sonra, alt süreç kapalıyken Drive'a hash'li checkpoint yazılır. İçerik adresli arşiv ve son başarılı arşiv işaretçisi kullanılır. Ani VM kaybında son kayıttan sonraki çalışma kaybolabilir. Eski arşivler tutulduğu için Drive alanı izlenmelidir.

Kurtarma hücresi arşiv hash'i ve güvenli dosya yollarını kontrol eder. Worker; aynı veri, kaynak, Python/paket, CUDA/sürücü ve GPU modeli koşulunu tekrar denetler. Farklı GPU veya ortam uyumsuzluğu otomatik giderilmez. Özel günlükler notebook çıktısına basılmaz. Sonuçlar yalnız `review/` altında denetim adayı olarak hazırlanır; CLI'nin sıfır çıkış kodu tek başına tamamlanma kanıtı değildir.

## Terminal / SSH ile benim yönetebilmem

[Google'ın resmî Colab CLI'si](https://github.com/googlecolab/google-colab-cli) oturum listeleme, dosya yükleme/indirme, kod çalıştırma ve SSH bağlantısını destekler. İlk kurulum resmî `v0.7.1` commit'i `c41c542629a1651292c834b7727615b9d5f9e5c2` ile yapılmıştı; bu tarihsel hazırlık adımı oturum açmamıştı. Şimdi ayrı `uv tool` ortamındaki CLI resmî `v0.7.4` commit'i `66d7963611bcce2d272488e982ad38e519c062f0` sürümüne güncellendi. Google OAuth girişi, oturum erişimi ve SSH bağlantısı doğrulandı. Güncel sürümün token yenileme ve kullanım sorgulama desteği operasyon yönetiminde kullanılabilir; kimlik bilgileri ve hesap kullanım ayrıntıları yayımlanmaz.

İlk CLI kod çalıştırma bağlantısı koptu; büyük SSH/SCP aktarımları başarısız oldu. Sonraki resmî `ContentsClient` aktarımı, 8 MiB boyutunda 40 parça bildirildikten sonra bağlantı hatasıyla durdu. Ardından sunucu oturum listesi boş döndü. Tam arşiv doğrulanmadı; bu parçalar tamamlanmış veri veya kalıcı yedek değildir. Kapanmanın bağlantı, CLI sürümü veya başka bir nedenden kaynaklandığı belirlenmedi. Tam dosya boyutu ve SHA-256, ardından güvenli çıkarma ve donuk veri kimliği doğrulanmadan benchmark başlatılmaz.

OAuth kodu, token veya SSH özel anahtarı sohbete/Git'e konmaz. Model erişimi gerekiyorsa ilgili Hugging Face koşulları ayrıca kabul edilmiş olmalıdır; Colab OAuth bunu sağlamaz.

Aşağıdaki `SESSION`, `colab sessions` çıktısından seçilen gerçek oturum adıdır:

```sh
colab status -s SESSION
colab ssh -s SESSION
colab download -s SESSION /content/rag-work/vidore-v3-physics-fr/state.json work/colab/physics-state.json
```

SSH komutu olmayan oturumu otomatik oluşturabilir; bağlanmadan önce gerçek oturum adı doğrulanır. CLI'nin doğrudan `upload` ve `download` işlemleri bütün dosyayı belleğe alır; büyük arşivler için doğrulanan parça aktarımı veya notebookun Drive `input/` yolu kullanılmalıdır. Ek abonelik veya işlem birimi satın alınmadı; açık GPU oturumları mevcut işlem birimi bakiyesini tüketir. Google girişi tamamlandı; bu bağlantı tek başına model veya benchmark başarısı anlamına gelmez.

Colab GPU türü, eşzamanlı kaynaklar ve çalışma süresi garanti edilmez. [Resmî kaynak sınırları](https://research.google.com/colaboratory/faq.html#resource-limits) geçerlidir; bu nedenle ölçüm yapmadan kaç kat hızlanacağını söylemiyoruz.

## Sonuç kabulü ve sonraki dalga

Worker'ın tam sorgu, 17 metrik, kanal ve 126 hücre kontrolü ilk kapıdır. Ardından özel SQLite sıralamaları, kaynak hash'leri, RRF, metrik formülleri ve mevcut koşullarla tekrar sayımı bağımsız denetlenir. Güven aralığı için kaynak bağımlılık yapısı incelenir; her koleksiyona aynı bootstrap varsayımı uygulanmaz. Yalnız güvenli toplamlar ve kaynak kod GitHub'a gider.

Finance FR ve HR'nin yerel kısmi cache'leri korunur. Bunları CUDA'ya taşıyarak sıfırdan aynı işi yaptığımızı iddia etmeyiz; sonraki aşamada cache uyumluluğu veya ayrı donanım karşılaştırması kararı gerekir. Rerankerler için aynı veri kümesini tek worker'a verme ve doğrulanmış arama cache'ini paylaşma planlanır. Gemma/QA ve BGE-extra CUDA desteği ayrıca doğrulanmadan bu notebooklara eklenmez. CIRR'nin resmî medya erişimi ve insanla kalibre edilmesi gereken değerlendirmeler GPU paralelliğiyle çözülmüş sayılmaz.

Hazırlık doğrulaması: 698 tam bağımlılıklı CPU testi, minimal ortamda 48 hedefli test ve Ruff geçti. Üç gerçek donuk veri kümesinin kimliği ve özel arşivlerinin hash/çıkarma doğrulaması geçti. Notebook kurulum ve veri taşıma kodu bağımsız incelendi; bu kontroller gerçek CUDA/model başarısı değildir.
