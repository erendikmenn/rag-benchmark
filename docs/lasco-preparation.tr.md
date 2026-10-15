# LaSCo: görsel ve metinle görsel arama hazırlığı

LaSCo, erişimi beklenen CIRR yerine yeni çalışmalar için seçilen **ayrı bir veri kümesidir**. Eski CIRR manifesti, sonuçları ve donmuş benchmark paydası değiştirilmez. Bu aşamada yalnız kaynaklar, temsiller ve PostgreSQL saklama yapıları hazırlanır; retrieval, reranker veya soru cevaplama değerlendirmesi çalıştırılmaz.

**15 Ekim 2026 kullanıcı durdurma talimatı:** Yerel açıklama üretimi, devam kuyruğu ve kendi Gemma sunucusu kapatıldı. Gemma E4B, projector ve eski Gemma 26B ağırlıkları önbellekteki hardlink/snapshot kopyalarıyla birlikte doğrudan silindi; Çöp Kutusu'na taşınmadı. Tekil model ağırlıkları toplamı 20.585.857.120 byte idi; ayrıca bu işe ait llama.cpp çalışma ortamı kaldırıldı. İşletim sisteminin erişim vermediği Çöp Kutusu'nun önceden var olan içeriği doğrulanamadı. BGE, EmbeddingGemma 2 ve diğer arama bağımlılıkları, hazırlanmış veriler, vektörler ve mevcut PostgreSQL veritabanı korundu. Bu durdurma anında başka bir model başlatılmamıştı. Sonraki kullanıcı talimatıyla aşağıdaki ayrı API hazırlığı yetkilendirildi.

Native N hazırlığı iki rolün tamamında bitti: 39.826 görsel + 30.037 sorgu = **69.863 vektör**, diskte korunuyor. Açıklamalarda **4.254 / 39.826** teknik kabul kaydı var; bir sonraki kesilmiş claim korunuyor ve otomatik tekrar yapılmıyor. Bunlar tam galeri veya insan tarafından doğrulanmış açıklamalar değildir. LaSCo G/E/J, BM25 ve yeni veritabanı aktarımı başlamadı; yeni LaSCo veritabanı yazımı ve benchmark sayısı sıfır. Aşağıdaki yerel üretim politikası tarihsel hazırlık kaydıdır, yeniden çalıştırma talimatı değildir.

**Sonraki API talimatı:** Kullanıcı Haiku 5.5 ile tam hazırlığı yetkilendirdi. İngilizce kaynak-only istemi, 512 çıktı token sınırı ve `google-vertex/global` rotasıyla en çok 100 eşzamanlı istek kullanıldı. İlk API üreticisi 342 kabulden sonra koptu; bu çıktılar tarihsel tutuldu ve yeni galeriye katılmadı. İkinci üretici 6.071 geçerli açıklama ve dört başarısız satırda kapandı. Bu 6.071 çıktı korunarak kalan satırlar işlendi; devam adımı toplam 39.810 başarı ve 16 eksikle kapandı. Ana devam adımının 1.409,16 saniyesi bütün hazırlığın süresi değildir. Kesilmiş veya normal bitiş üretmeyen cevaplar kabul edilmez. Önceki ücretler ve kullanım bilgisi gelmeyen denemelerin üst sınırları dahil bütçe 15 USD’dir. [Güvenli API durum kaydı](../reports/lasco-haiku-api-preparation.json).

Ana geçiş ve eski koordinatör kapandıktan sonra kalan 16 kayıt, aynı Haiku 5.5 modelinin açık `anthropic` rotasında 16 denemeyle tamamlandı. İlk kanarya başarılıydı; 11 sağlayıcı hatalı kayıtta özgün istem korundu, gerçek `length` bitişi doğrulanmış beş kayıtta aynı 512 token sınırı altında en çok 80 İngilizce kelime isteyen genel sistem eki kullanıldı. Önceki 39.810 başarılı kayıt değişmedi. Böylece **39.826 / 39.826 açıklama normal bitişle kabul edildi**. Sağlayıcı ve koşullu istem farkları satır bazında kaydedilir; karma çıktı değişmemiş tek bir Google politikası olarak sunulmaz. Toplam bildirilmiş üretim kullanımı 6,2381572 USD; bilinmeyen denemelerin üst sınırları dahil ihtiyatlı toplam 6,3489028 USD’dir. Bu tutarlar pilot ölçümünü içermez ve fatura değildir. [Tam bağımsız kaynak/cevap/maliyet/türetilmiş veri denetimi](../reports/lasco-haiku-caption-independent-audit.json) 39.826 açıklama ve 30.037 sorguda geçti. Sonraki embedding yapılandırması adımı model önbelleğindeki bir dosya bağlantısı kontrolünde durdu; düzeltme hazırlanıyor. G/E/J henüz başlamadı; BM25, PostgreSQL ve gerçek dump/restore doğrulaması da bekliyor. Normal bitiş insan tarafından ölçülmüş anlam doğruluğu değildir.

Açıklama galerisi tam ve bağımsız denetlenmiş olduğunda G/E/J belge+sorgu vektörleri, BM25 ve PostgreSQL aktarımı devam eder. Native N tekrar üretilmez. Nihai hedef 28 görünüm, 184 multimodal set ve RAGTurk dahil 1.128.125 dense vektördür; şu anda bu hedef tamamlanmış değildir. Kaynak/vektör geri okuma ve gerçek dump/restore karşılaştırması bitmeden veritabanı hazır sayılmaz. Yeni retrieval, reranker, QA veya benchmark değerlendirmesi başlatılmaz.

## Kaynak ve tam kapsam

[Resmî LaSCo deposunun](https://github.com/levymsn/LaSCo) `311f4d7d4bc069fc2d8abd3bf523cdfae6b5800f` sürümü kullanılır. Validation bölümünde **39.826 galeri görseli ve 30.037 hedefi belirtilmiş sorgu** vardır. Her sorgu bir referans görsel ile değişiklik talimatından oluşur; hedef görsel yalnız ayrı qrel kaydındadır. Referans görsel sonraki aramada adaylardan dışlanır.

Resmî dosyadaki altı boş/yalnız boşluk içeren değişiklik metni aynen korunur ve `transition_text_empty` ile işaretlenir. Bunlar görsel girdisi olan, değişiklik metni olmayan sorgulardır. 10.133 negatif sorgu kimliği de resmî ters yönlü kayıtlara aittir; negatif oldukları için silinmez. LaSCo'nun metinleri VQA2.0 çiftlerinden otomatik/GPT-3 ile türetilmiştir; tamamen insan yazımı göreli açıklamalar olarak sunulmaz.

Galeri resmî COCO `val2014.zip` dosyasından alınır. Arşivin ilk gözlenen SHA-256 değeri yerel aktarım kimliğidir; yayıncının yayımladığı checksum kanıtı değildir. Kullanılan her ZIP üyesinin CRC'si, boyutu, dosya türü ve güvenli yolu; ardından bütün kaynak/medya SHA-256 değerleri doğrulanır. ZIP'in geri kalan görselleri LaSCo galerisine eklenmez. LaSCo anotasyonlarının MIT lisansı, COCO görsellerinin kendi lisanslarının yerine geçmez.

## Planlanan temsiller

| Kanal | Galeri girdisi | Sorgu girdisi |
|---|---|---|
| N / native EG2 | Görsel | Referans görsel + özgün değişiklik metni |
| G / BGE-M3 | Yalnız kaynak görselden üretilen açıklama | Referansın açıklaması + özgün değişiklik metni |
| E / EG2 metin | Aynı kaynak açıklaması | Aynı referans açıklaması + değişiklik metni |
| J / EG2 birleşik | Görsel + kaynak açıklaması | Referans görsel + özgün değişiklik metni |
| B / BM25 | Aynı kaynak açıklaması | Aynı referans açıklaması + değişiklik metni |

G/E/B için hedef görselin etiketi veya hedefe özel açıklama sorguya eklenmez. Galeri açıklaması sorgulardan bağımsız, yalnız piksel içeriğiyle üretilir. Native N hazırlığı açıklama beklemez. N özgün görünümde; G/E/J ve BM25 ayrı türetilmiş görünümde saklanır. Dört yoğun kanalın iki rolü toplam **8 set / 279.452 dense vektör** planlar; BM25 ayrı bir lexical indekstir. Bu sayılar hazırlık tamamlanma kanıtı veya benchmark sonucu değildir.

## Çevrimdışı içe aktarma

`python -m rag_benchmark.lasco_data download --raw-dir ...` yalnız açıkça istenen indirmeyi yapar. Ağdan bağımsız `import` komutu, pinli iki anotasyon dosyası ve SHA-256 değeri belirtilen yerel COCO arşivi ister. Tam galeri/sorgu sayıları, her referans ve hedefin galeride bulunması, referansın hedef olmaması ve tekil kimlikler zorunludur. Dizin yayını atomiktir ve mevcut hedefi değiştiremez. Yarım hazırlık tam veri olarak açılmaz.

Kaynaklar, model çıktıları ve özel bağlantı bilgileri GitHub'a yüklenmez. Mevcut 26 görünüm / 176 set ve doğrulanmış yedek korunur. Yeni hazırlık ancak tam embedding kapsamı, PostgreSQL'den bütün değerlerin geri okunması ve yeni yedek doğrulaması tamamlandıktan sonra tamamlanmış sayılır.

## Doğrulanan kaynak ve açıklama hazırlığı

Tam arşiv aktarımı ve bağımsız kaynak denetimi tamamlandı: 39.826 farklı görsel byte hash'i, 30.037 sorgu ve 30.037 hedef ilişkisi doğrulandı. Bütün görseller tam açılıp RGB olarak okundu. Resmî arşivde `.jpg` adıyla bulunan bir dosyanın içeriği PNG, kalan 39.825 dosya JPEG'dir; özgün byte'lar değiştirilmez. [Güvenli denetim özeti](../reports/lasco-source-preparation.json). Native embedding üretimi tamamlanmıştır; yeni temsillerin PostgreSQL aktarımı henüz tamamlanmamıştır.

`scripts/prepare_composed_captions.py` önce açık bir hazırlık dosyası oluşturur; ayrı `--execute` çağrısı dosyanın dış SHA-256 kimliğini ister. Yalnız kaynak görsellerinden İngilizce açıklama üretir: E4B Q4, 512 çıktı tokenı, temperature 0, seed 42, thinking off, image1120 ve repeat_penalty 1.1. Bu ayrı politikadır. Model, projector, sunucu, kaynak ve runtime kimlikleri doğrulanır; GPU kilidi kendi sunucusunun kapanmasına kadar tutulur. Ham yanıt kabul denetiminden önce özel alana kaydedilir. Normal bitiş, token/context sınırı ve tekrarlama kontrolü geçmeyen kayıt tamamlanmış sayılmaz; otomatik tekrar veya çıktı kırpma yoktur.

Kısmi açıklamalar tam galeri olarak yayımlanmaz. Tamamlanan görünüm özgün sorguları, referans dışlamasını ve hedef ilişkilerini korur; hedef bilgisi üretim isteğine girmez. Hazırlık kodunun CPU testleri gerçek caption anlam doğruluğu veya benchmark başarısı değildir.
