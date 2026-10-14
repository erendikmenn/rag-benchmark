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

Hazırlık ve aktarım sürmektedir. Resmî medyası bulunmayan CIRR ile tamamlanmamış kaynak açıklaması temsilleri tam olarak işaretlenmez. Yerel önbellek, PostgreSQL'e aktarılmış ve doğrulanmış kayıt yerine geçmez. Nihai kapsam ve doğrulama sayıları ancak gerçek aktarım ve bağımsız okuma tamamlanınca yayımlanacaktır.
