# Çok modlu veri ve embedding hazırlığı

Bu aşama yalnızca veri, embedding ve BM25 indekslerinin hazırlanması ve saklanması içindir. Arama, reranker, soru cevaplama ve benchmark değerlendirmesi başlatılmaz. Tamamlanmış RAGTurk hazırlığı `ragturk_prepared` şemasında korunur; diğer koleksiyonlar `multimodal_prepared` şemasına eklenir.

PostgreSQL, `compose.ragturk-pgvector.yml` içindeki özel serviste çalışır. Veriler Docker imajına gömülmez; PostgreSQL'in kalıcı named volume'ünde tutulur. Başka makineye taşımak için ayrıca doğrulanmış veritabanı yedeği gerekir. Bir volume tek başına başka makinede otomatik oluşmaz.

## Saklanan yapılar

| Yapı | Saklama biçimi |
|---|---|
| BGE, EmbeddingGemma, native medya ve yoğun specialist vektörleri | Kaynak/rol/model kimliği ve boyutuyla ayrı pgvector kayıtları |
| ColQwen PDF temsili | Her sayfa/sorgu için sıralı, değişken uzunlukta FP32 token matrisi; ortalama alınmaz |
| JavaScript ortak parçalama | Fonksiyon → sıralı parça eşlemesi, karakter sınırları ve kaynak hash'i korunur |
| BM25 | Özgün bm25s CSC indeks dosyaları, sözcük sözlüğü, belge sırası ve uzunlukları |
| Kaynak ve medya | SHA-256 ile adreslenen, boyutu sınırlı bytea parçaları; aynı içerik tekrar saklanmaz |
| Değerlendirme etiketleri | Ayrı etiket referansları; encoder girdisi yapılmaz |

Her embedding kümesinin veri, model, sürüm, cihaz, hassasiyet, ön işleme ve rol kimliği bulunur. CUDA ve MPS çıktıları birbirinin sonucu olarak yeniden adlandırılmaz. Başka bir model eklendiğinde mevcut kümeyi değiştirmek yerine yeni bir küme eklenir. Kaynak veri değişikliği de yeni veri kimliği gerektirir.

## Hazırlama araçları

- `scripts/prepare_multimodal_embeddings.py`: İncelenmiş özel iş tanımındaki G/E/N/J temsillerinin yalnızca eksik, 128 satırlık bloklarını üretir. Mevcut blokların hash'leri ve gerçek adapter kimliği uyuşmalıdır. Başarısız veya yarım kalan çalışma otomatik yeniden denenmez.
- `scripts/prepare_multimodal_bm25.py`: Kaynak metin üzerinde mevcut benchmark'ın tokenizasyonu ve BM25 ayarlarıyla yalnız indeks oluşturur. Sorgu skoru veya sıralama üretmez.
- `scripts/prepare_multimodal_colqwen.py`: PDF sayfaları ve sorguları için özgün 128 boyutlu token matrislerini üretir; MaxSim araması yapmaz. Önceden mühürlenen bir dosyanın silinmesi yeniden çıkarım izni sayılmaz. Yarım kalan bir işi sürdürmek açık resume ve aynı kimlik gerektirir.
- `scripts/multimodal_pgvector_store.py`: Özel manifestten kayıpsız aktarım ve salt okunur doğrulama yapar. Mevcut tam kümeyi değiştirmez; yeni model kümeleri eklenebilir.

Bağlantı bilgileri ve veri taşıyan manifestler ignored özel dizinde tutulur. GitHub'a kaynak metin, medya, sorular, cevaplar, vektör verisi veya kimlik bilgileri yüklenmez.

## Doğrulama ve sınırlar

Kaynak kimlikleri, sıralı kayıt eşlemeleri, tam satır sayıları, vektör boyutları, sonlu değerler, normalizasyon ve dosya hash'leri kontrol edilir. PostgreSQL'e yazılan bütün FP32 değerleri ve dosya byte'ları geri okunup karşılaştırılır. Ragged matrislerde token sırası ve matris şekli de korunur. Ayrı denetçi aynı aktarım fonksiyonlarını kullanmadan SQL'den okuyarak karşılaştırabilir.

Bu kontroller saklama ve kaynak eşlemesinin doğruluğunu sınar; embedding modelinin anlamsal başarısı veya sonraki benchmark skoru değildir. Geçmiş önbelleğe bugün eklenen hash, geçmişte kaydedilmiş bir checksum kanıtı sayılmaz.

**Hazırlık ve saklama:** 26 örtüşen görünüm / 176 multimodal set; RAGTurk dahil **848.673 dense vektör**, 21 native BM25 indeksi ve ayrı 21.671 ragged öğe matrisi. İki fotoğraf açıklaması görünümü ve yeni veritabanı temsilleri doğrulamadan geçti; CIRR resmî medyası hâlâ eksik. Benchmark 2.052 / 18.460 düzeyinde duraklatıldı; yeni retrieval veya QA değerlendirmesi yok. Yeni yedek gerçek geri yükleme ve 12 tablonun tam byte/katalog karşılaştırmasını geçti; geçici doğrulama veritabanı kaldırıldı. [Kapsam, sayılar ve yedek kanıtı](#çalışan-veritabanı-ve-yedek).

21.671 ColQwen öğe matrisi toplam 14.559.743 adet 128 boyutlu token vektörü içerir; matris ve token vektörü sayıları ayrı tutulur. Hazırlanmış temsil bulunması geçmiş başarısız veya yarım benchmark çalışmasını tamamlanmış yapmaz. Opsiyonel tarihsel çalışma ortamı varyantları ayrıca eksik kalabilir; CUDA ve MPS kimlikleri korunur. Yeni güven aralığı veya semantik doğruluk ölçümü üretilmedi.

## Doğrulanan kapsam — 15 Ekim 2026

| Kaynak türü | Görünüm | Temsil seti | Dense vektör | Ragged öğe matrisi |
|---|---:|---:|---:|---:|
| Kod | 6 | 24 | 471.918 | 0 |
| Belge | 8 | 80 | 86.684 | 21.671 |
| Çevresel ses | 4 | 28 | 58.464 | 0 |
| Fotoğraf | 4 | 20 | 108.165 | 0 |
| Konuşma | 2 | 10 | 5.360 | 0 |
| Video | 2 | 14 | 14.000 | 0 |
| Multimodal toplam | 26 | 176 | 744.591 | 21.671 |
| RAGTurk (ayrı) | — | — | 104.082 | 0 |

Özgün ve türetilmiş görünümler kaynak kayıtlarını veya medyayı paylaşabilir; 26 görünüm, 26 bağımsız veri kümesi anlamına gelmez. Dense satırlar ile değişken uzunluklu öğe matrisleri ayrı sayılır. ColQwen matrisleri öğe başına birden çok token satırı içerir; matris sayısı token sayısı değildir. [Güvenli hazırlık özeti](../reports/multimodal-pgvector-preparation.json) yalnız toplu kapsam ve doğrulama bilgilerini yayımlar.

## Çalışan veritabanı ve yedek

15 Ekim’de eksik fotoğraf açıklamalarının kurtarılması için, ilk takılan Türkçe görüntüde tek kontrollü çağrı tamamlandı. Aynı kaynak, istem ve 512 token sınırı korunup yalnız `repeat_penalty=1.1` eklendi; çıktı 95 token ile normal bitti ve ardışık aynı kelime tekrarı görülmedi. Bağımsız denetim, 1.446 eski geçerli açıklamanın değişmediğini doğruladı. Bu [tek kaynak teknik kontrolü](../reports/multimodal-photo-caption-recovery-health.json), tam fotoğraf açıklaması veya yeni veritabanı kapsamı sayılmaz. Ardından Türkçe görünümün **3.600/3.600** kaynak açıklaması tamamlandı ve bağımsız denetimden geçti: 1.446 eski kayıt değişmeden korundu; 2.154 yeni politika kaydının biri önceki teknik kontrolden alındı, kalan 2.153 kaynak için yeni çağrı yapıldı. Satırların üretim politikaları ayrı tutulur. [Tam Türkçe hazırlık kanıtı](../reports/multimodal-photo-caption-preparation.json) kaynak, medya ve sorgu/alaka eşlemelerini doğrular; anlamsal açıklama doğruluğu ölçümü değildir. İngilizce görünümün de **3.600/3.600** açıklaması tamamlandı ve bağımsız denetimden geçti; İngilizce politika altında 3.600 yeni kaynak çağrısı yapıldı. Türkçe ve İngilizce G/E/J embedding’leri ile iki native BM25 indeksi artık hazır; additive PostgreSQL aktarımı, store verify ve 176 setin bağımsız tam geri okuması geçti. Bu nedenle güncel veritabanı sayıları yukarıdaki 26/176 kapsamına yükseldi. Yeni yedek ayrı bir veritabanına geri yüklendi; 12 tablonun bütün ikili içeriği ve kayıtlı katalog bilgileri kaynakla birebir eşleşti. Son kaynak kontrolü de değişmedi; yalnız bu doğrulama için oluşturulan veritabanı kaldırıldı. CIRR’nin resmî görüntü erişimi hâlâ eksiktir.

Hizmet `ragturk-prepared-db-1`, kalıcı volume `ragturk-prepared_ragturk_pgvector_data` ve yerel bağlantı `127.0.0.1:55433` kullanılır. Sabit PostgreSQL/pgvector imajı compose dosyasında kayıtlıdır. Model ve veri sürümleri [temsil envanterinde](../reports/multimodal-representation-inventory.json) ayrı tutulur.

**Yeni 26/176 kapsamının yedeği:** `prepared-photo-75c54a33f79ff09b-ff25f1e5f8760cb2.dump`; `22854578574` bayt; SHA-256 `6d470a9a8175fd41f256945c2b373959e546c311c74f0906bc5cac1e7b877256`. Gerçek geri yükleme tamamlandı: iki şemadaki 12 tablonun bütün ikili baytları ve kaydedilen katalog bilgileri eşleşti. Kaynak veritabanı son kez tekrar kontrol edildi ve değişmedi. Yalnız bize ait geçici doğrulama veritabanı ve onun özel bağlantı dosyası kaldırıldı. Roller ve erişim izinleri bağımsız tablo/katalog karşılaştırmasının dışındadır. Arşiv ve bağlantı bilgileri GitHub’a gönderilmez. Önceki 24/164 kapsamının doğrulanmış yedeği ve restore kanıtı tarihsel kayıt olarak değiştirilmeden korunur.

Fotoğraf kanal ayrımı: native N görüntüyü doğrudan EG2 ile, S görüntü specialist’i ile temsil eder; açıklama bu iki kanal için zorunlu değildir. Yeni G/E kaynak açıklaması metnini, J görüntü+açıklamayı kullanır; B/BM25 açıklama metnini indeksler. Türkçe satırlar 1.446 eski +2.154 yeni politika olarak ayrı provenans taşır; İngilizce 3.600 yeni politika satırıdır. Güncel manifest SHA-256: `75c54a33f79ff09b90574b51f04af984e5e5502ecf5f93d63336fea47f439286`. 88 belge/sorgu temsil çifti hazırlanmıştır; bu sayılar semantik doğruluk veya yeni benchmark sonucu değildir.
