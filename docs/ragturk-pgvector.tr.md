# RAGTurk: PostgreSQL + pgvector hazırlığı

Bu hazırlık 37.511 kaynak parçasını ve **14.530 sorunun tamamını** kapsar:
2.000 geliştirme ve 12.530 test sorusu. Split etiketleri korunur. Soruların
embedding'lerini hazırlamak, benchmark çalıştırmak veya cevap üretmek değildir.

## Saklanan veriler

`ragturk_prepared` şemasında:

| Tablo | İçerik |
|---|---|
| `chunks` | Kaynak metin, başlık, kaynak/belge kimliği, sıra; BGE `vector(1024)` ve EmbeddingGemma 2 `vector(768)` |
| `questions` | Soru metni, split, sıra ve kaynak bilgileri; iki modelin soru vektörleri |
| `question_labels` | Soruya bağlı referans cevaplar ve doğru chunk kimlikleri |
| `files` | Dört kaynak dosyasının ve altı özgün BM25 indeks dosyasının byte'ları, SHA-256 ve boyutları |
| `bundle` | Hazırlık kimliği, kaynak/model/üretim kaydı ve transaction tamamlanma durumu |

Toplam **104.082 dense vektör**: her model için 37.511 belge + 14.530 soru.
Belge vektörlerinin doğrulanmış mevcut MPS FP32 önbellekleri kullanılır. Soru
vektörleri aynı model revizyonları ve özgün sorgu önekleriyle yeni üretilir.
Referans cevaplar ve doğru kaynak etiketleri embedding girdisi değildir.
Etiketlerin ayrı tabloda olması tek başına erişim yetkisi izolasyonu sağlamaz;
gelecekteki deney işçilerine ayrıca uygun roller verilmelidir.

BM25, özgün `bm25s` indeksinin altı dosyasıyla **PostgreSQL içinde saklanır**.
PostgreSQL `ts_rank` veya başka bir BM25 formülüyle değiştirilmez. Bu hazırlık
SQL içinde BM25 arama motoru uygulamaz; gelecekteki işçi, saklanan dosyaları
hash denetimiyle çıkarıp aynı `bm25s` sürümüyle açabilir. Şimdilik herhangi bir
arama veya sıralama çalıştırılmaz. Yaklaşık vektör arama indeksi oluşturulmaz.

## Yerel servis ve kalıcılık

Servis `compose.ragturk-pgvector.yml` ile ayrı Docker projesinde çalışır:

```sh
docker compose -f compose.ragturk-pgvector.yml -p ragturk-prepared up -d
```

Bağlantı `127.0.0.1:55433`, veritabanı `ragturk`, sahibi `ragturk_owner`.
Özel parola ve bağlantı bilgisi gitignored `work/ragturk-pgvector/` altındadır:
`postgres-password` ve `connection.json`. Bu dosyalar GitHub'a yüklenmez.
Compose çalışmadan önce özel parola dosyası bulunmalıdır. Port yalnız loopback'e
açılır. Diğer projelerin PostgreSQL servisleri kullanılmaz.

Veri Docker imajının içinde değil, `ragturk-prepared_ragturk_pgvector_data`
kalıcı volume'ündedir. Konteyner yeniden oluşturulsa da volume korunur.
`docker compose down -v` veriyi siler; normal durdurma için `stop` kullanılır.
Başka makineye taşıma için özel `pg_dump` arşivi ve SHA-256 kullanılır.

İlk, yalnız RAGTurk hazırlığının tarihsel doğrulanmış yedeği özel dizindeki
`ragturk-c4c3142416428e71.dump` dosyasıdır; checksum `backup.sha256` içinde.
Arşiv kataloğu okunup beş tablonun veri bölümleri doğrulandı; ayrı bir geri
yükleme provası o tarihsel yedek için yapılmadı. Daha sonra iki hazırlık şemasını kapsayan özel final yedek gerçek ayrı veritabanına geri yüklendi; kaynak ve geri yüklenen veritabanındaki bütün tablo ikili baytları ve kaydedilen tablo kataloğu eşleşti. Roller/izinler bu iddianın dışında. Tarihsel yedek kaydı bu yeni karşılaştırmayla geriye dönük değiştirilmez. [Güvenli hazırlık özeti](../reports/ragturk-pgvector-preparation.json)
yalnız sayıları, sürümleri ve hash'leri içerir.

## Hazırlık komutları

İsteğe bağlı PostgreSQL istemcisi `postgres` extra'sındadır. Mevcut model
ortamını değiştirmeden yalnız istemciyi eklemek için:

```sh
uv pip install --python .venv/bin/python 'psycopg[binary]>=3.3,<4'
```

Sorgu hazırlığı tek GPU sahibiyle modelleri sırayla yürütür. `--execute`
verilmezse yalnız kaynak ve konfigürasyon denetlenir:

```sh
.venv/bin/python scripts/prepare_ragturk_queries.py --model bge \
  --output work/ragturk-pgvector/bge --execute
.venv/bin/python scripts/prepare_ragturk_queries.py --model embeddinggemma \
  --output work/ragturk-pgvector/embeddinggemma --execute
```

Her 128 soruluk blok kimlik ve hash ile korunur; model batch'i 32'dir.
Başarılı bloklar aynı kimlikle yeniden kullanılabilir. Bu toplu sorgu üretimi,
tarihsel tek-sorgu çağrılarıyla bit düzeyinde eşdeğerlik iddiası taşımaz.
Veri veya chunk/model/tokenizer/prompt politikası değişirse aynı cache kimliği
yeniden kullanılmamalıdır. Belge önbelleklerinin SHA'ları bugün doğrulanan
byte'ları gösterir; geçmişte tutulmamış hash kaydının yerini tutmaz.

Özel `prepared-bundle.json`, tüm dosyaların yollarını/SHA'larını, sıra
kimliklerini ve iki modelin kökenini bağlar. İçe aktarma komutları:

```sh
.venv/bin/python scripts/ragturk_pgvector_store.py validate \
  --bundle work/ragturk-pgvector/prepared-bundle.json
# RAGTURK_PG_DSN özel bağlantı dosyasından süreç ortamına alınır; loglanmaz.
.venv/bin/python scripts/ragturk_pgvector_store.py import \
  --bundle work/ragturk-pgvector/prepared-bundle.json
.venv/bin/python scripts/ragturk_pgvector_store.py verify \
  --bundle work/ragturk-pgvector/prepared-bundle.json
```

İçe aktarma tek transaction'dır. Tam sayılar, metinler, etiketler, dört vektör
kümesinin FP32 değerleri ve BM25 dosyaları okunup doğrulanmadan tamamlanmaz.
Aynı tamamlanmış paket tekrar verilirse yalnız doğrulanır; başka bir kimlik
mevcut şemanın üzerine yazılmaz. `verify` depolama bütünlüğü kontrolüdür;
retrieval/QA başarı ölçümü değildir.

Ham veriler, referans cevaplar, özel paket manifesti, günlükler ve DB yedeği
özel kalır. RAGTurk kaynağının CC-BY-NC-SA-4.0 lisansı korunur.
