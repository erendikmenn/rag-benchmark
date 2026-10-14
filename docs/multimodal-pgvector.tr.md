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

**Desteklenen hazırlık/saklama kapsamı eksikler açık tutularak doğrulandı:** 24 örtüşen frozen görünümde 164 multimodal temsil seti bulunuyor: 679.692 dense vektör ve 21.671 değişken uzunluklu öğe matrisi. Ayrı RAGTurk hazırlığındaki 104.082 dense vektörle iki şemanın toplamı 783.774 dense vektördür. On sekiz multimodal native BM25 indeksi ve ayrı RAGTurk indeksi PostgreSQL kalıcı Docker volume’ünde saklanıyor. Bütün kaynak/vektörlerin bağımsız geri okuması ve özel yedeğin gerçek geri yükleme karşılaştırması, iki şemadaki tablo ikili baytlarını ve kaydedilen tablo kataloğunu doğruladı; roller ve izinler bu geri yükleme iddiasının dışında. Görünümler bağımsız veri kümeleri değildir. Fotoğrafın native N/S temsilleri mevcut; yalnız kaynak metni kullanan B/G/E/J dalları ve CIRR resmî medyası eksik. Yeni arama, reranker, cevap üretimi veya semantik doğruluk sonucu iddia edilmiyor; benchmark duraklatılmış durumda ve ana kapsam 2.052 / 18.460 olarak kaldı.

21.671 ColQwen öğe matrisi toplam 14.559.743 adet 128 boyutlu token vektörü içerir; matris ve token vektörü sayıları ayrı tutulur. Hazırlanmış temsil bulunması geçmiş başarısız veya yarım benchmark çalışmasını tamamlanmış yapmaz. Opsiyonel tarihsel çalışma ortamı varyantları ayrıca eksik kalabilir; CUDA ve MPS kimlikleri korunur. Yeni güven aralığı veya semantik doğruluk ölçümü üretilmedi.

## Doğrulanan kapsam — 15 Ekim 2026

| Kaynak türü | Görünüm | Temsil seti | Dense vektör | Ragged öğe matrisi |
|---|---:|---:|---:|---:|
| Kod | 6 | 24 | 471.918 | 0 |
| Belge | 8 | 80 | 86.684 | 21.671 |
| Çevresel ses | 4 | 28 | 58.464 | 0 |
| Fotoğraf | 2 | 8 | 43.266 | 0 |
| Konuşma | 2 | 10 | 5.360 | 0 |
| Video | 2 | 14 | 14.000 | 0 |
| Multimodal toplam | 24 | 164 | 679.692 | 21.671 |
| RAGTurk (ayrı) | — | — | 104.082 | 0 |

Özgün ve türetilmiş görünümler kaynak kayıtlarını veya medyayı paylaşabilir; 24 görünüm, 24 bağımsız veri kümesi anlamına gelmez. Dense satırlar ile değişken uzunluklu öğe matrisleri ayrı sayılır. ColQwen matrisleri öğe başına birden çok token satırı içerir; matris sayısı token sayısı değildir. [Güvenli hazırlık özeti](../reports/multimodal-pgvector-preparation.json) yalnız toplu kapsam ve doğrulama bilgilerini yayımlar.

## Çalışan veritabanı ve yedek

Hizmet `ragturk-prepared-db-1`, kalıcı volume `ragturk-prepared_ragturk_pgvector_data` ve yerel bağlantı `127.0.0.1:55433` kullanılır. Sabit PostgreSQL/pgvector imajı compose dosyasında kayıtlıdır. Model ve veri sürümleri [temsil envanterinde](../reports/multimodal-representation-inventory.json) ayrı tutulur.

İki şemayı kapsayan özel yedek `work/multimodal-pgvector/prepared-all-b1a5170f3b8ad0e7.dump` dosyasıdır: 22.625.348.542 bayt. SHA-256 ve doğrulama bağları [hazırlık raporundadır](../reports/multimodal-pgvector-preparation.json). Yedek ayrı bir veritabanına gerçekten geri yüklendi; 12 tablonun bütün ikili verileri ve kaydedilen tablo kataloğu kaynakla eşleşti. Geçici doğrulama veritabanı kaldırıldı. Roller ve erişim izinleri bu bağımsız karşılaştırmanın kapsamına dahil değildir. Arşiv ve bağlantı bilgileri GitHub’a gönderilmez.
