# Google Colab ile paralel devam

14 Ekim 2026: ilk üç notebook ve özel veri paketleri hazırlandı. **Gerçek Colab modeli veya GPU deneyi henüz çalıştırılmadı.** Mac'teki eski kuyruk ve silinen otomasyon yeniden başlatılmadı. Yayımlanmış ana kapsam 2.052 / 18.460 olarak kalır.

## İlk dalga

Her notebook ayrı bir Colab çalışma zamanında bir tam veri kümesi işler. İlk dalga, yerelde tam dense araması yapılmamış üç koleksiyonu kullanır. Bu oturum sayısı hedefidir; hesapta gerçekten açılabilecek eşzamanlı GPU sayısı doğrulanmadan kaynak taahhüt edilmez.

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

[Google'ın resmî Colab CLI'si](https://github.com/googlecolab/google-colab-cli) oturum listeleme, dosya yükleme/indirme, kod çalıştırma ve SSH bağlantısını destekler. Mac'e ayrı bir `uv tool` ortamında, resmî `v0.7.1` commit'i `c41c542629a1651292c834b7727615b9d5f9e5c2` üzerinden kuruldu; `colab version` ve `colab ssh --help` doğrulandı. Bu kurulum Colab oturumu açmadı.

İlk Google girişini Terminal'de tamamlayın:

```sh
/Users/mehmeterendikmen/.local/bin/colab sessions
```

Tarayıcıdaki Google girişinden sonra istenen kodu yalnız bu Terminal'e girin. Giriş tamamlanınca mevcut oturum adlarını okuyabilir, donanım/kota bilgisine göre iş atayabilir ve sonuçları bu sohbetten yönetebilirim. OAuth kodu, token veya SSH özel anahtarı sohbete/Git'e konmaz.

Aşağıdaki `SESSION`, `colab sessions` çıktısından seçilen gerçek oturum adıdır:

```sh
colab status -s SESSION
colab ssh -s SESSION
colab upload -s SESSION work/colab/bundles/vidore-v3-physics-fr.tar /content/vidore-v3-physics-fr.tar
colab download -s SESSION /content/rag-work/vidore-v3-physics-fr/state.json work/colab/physics-state.json
```

SSH komutu olmayan oturumu otomatik oluşturabilir; bağlanmadan önce gerçek oturum adı doğrulanır. Yukarıdaki doğrudan yükleme SSH/CLI akışı içindir; notebookun varsayılanı Drive `input/` yoludur. Bu iki yöntemin giriş yolları açıkça eşleştirilmelidir. Ücretli kaynak veya plan satın alınmadı. Google girişini tamamlamak, kullanılabilir oturumların okunması için gereken bağlantı adımıdır.

Colab GPU türü, eşzamanlı kaynaklar ve çalışma süresi garanti edilmez. [Resmî kaynak sınırları](https://research.google.com/colaboratory/faq.html#resource-limits) geçerlidir; bu nedenle ölçüm yapmadan kaç kat hızlanacağını söylemiyoruz.

## Sonuç kabulü ve sonraki dalga

Worker'ın tam sorgu, 17 metrik, kanal ve 126 hücre kontrolü ilk kapıdır. Ardından özel SQLite sıralamaları, kaynak hash'leri, RRF, metrik formülleri ve mevcut koşullarla tekrar sayımı bağımsız denetlenir. Güven aralığı için kaynak bağımlılık yapısı incelenir; her koleksiyona aynı bootstrap varsayımı uygulanmaz. Yalnız güvenli toplamlar ve kaynak kod GitHub'a gider.

Finance FR ve HR'nin yerel kısmi cache'leri korunur. Bunları CUDA'ya taşıyarak sıfırdan aynı işi yaptığımızı iddia etmeyiz; sonraki aşamada cache uyumluluğu veya ayrı donanım karşılaştırması kararı gerekir. Rerankerler için aynı veri kümesini tek worker'a verme ve doğrulanmış arama cache'ini paylaşma planlanır. Gemma/QA ve BGE-extra CUDA desteği ayrıca doğrulanmadan bu notebooklara eklenmez. CIRR'nin resmî medya erişimi ve insanla kalibre edilmesi gereken değerlendirmeler GPU paralelliğiyle çözülmüş sayılmaz.

Hazırlık doğrulaması: 698 tam bağımlılıklı CPU testi, minimal ortamda 48 hedefli test ve Ruff geçti. Üç gerçek donuk veri kümesinin kimliği ve özel arşivlerinin hash/çıkarma doğrulaması geçti. Notebook kurulum ve veri taşıma kodu bağımsız incelendi; bu kontroller gerçek CUDA/model başarısı değildir.
