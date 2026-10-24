# ViDoRe Industrial: Laya’lı ve Laya’sız belge erişimi

İngilizce Industrial koleksiyonunun **5.244 sayfası ve 283 yazılı sorgusu** için gerçek çıktılar ve bağımsız bilimsel inceleme tamamlandı. **126 Laya’sız (None) koşul, 378 Laya koşuluyla** aynı sorgu grubu, kanal alt kümesi ve bütçede eşleştirildi. Bu koleksiyon, [daha önce yayımlanan altı koleksiyondan](vidore-laya-all-six-results-v2.tr.md) ayrıdır. Sonuçlar konuşulan sorgu veya geniş tüm-modalite başarımı olarak yorumlanmaz.

B = BM25, G = BGE-M3 metin, E = EG2 metin, N = EG2 ham sayfa görseli, S = ColQwen2.5, J = EG2 ortak metin + görsel. Altı kanalın 63 boş olmayan alt kümesi, iki bütçe ve Laya K=20/50/100 kullanıldı. Bütçe 100, RRF sabiti 60’tır. `per_channel` her kanaldan 100; `total` kanallar arasında paylaştırılmış toplam 100 aday alır.

Hit@K ilk K içinde en az bir ilgili sayfa bulunan sorguların oranıdır. Recall@K ilgili sayfaların ilk K içinde bulunan payıdır. Aşağıdaki oranlar yüzde, farklar yüzde puandır.

## Sabit karşılaştırma: B+G+E+N+S+J, Laya K=50

Ana ayar `per_channel`, K=50’dir; `total` ayrıca gösterilir. Tablo en iyi koşullar seçilerek oluşturulmadı. None elde edilen aday birleşimini sıralar; Laya ilk K adayı yeniden sıralar. Farklar aday kesimini ve yeniden sıralamayı birlikte içerir.

| Bütçe | Yöntem | Hit@5 | Hit@20 | Recall@5 | Recall@20 |
|---|---|---:|---:|---:|---:|
| per_channel | None | 74.56% | 88.69% | 46.16% | 62.58% |
| per_channel | Laya K=50 | 42.40% | 78.45% | 19.61% | 46.92% |
| per_channel | Laya − None (yüzde puan) | -32.16 | -10.25 | -26.54 | -15.65 |
| total | None | 77.39% | 91.87% | 46.82% | 64.40% |
| total | Laya K=50 | 43.82% | 80.92% | 19.85% | 50.57% |
| total | Laya − None (yüzde puan) | -33.57 | -10.95 | -26.97 | -13.83 |

Bu sabit örnekte Laya dört ana ölçütte de daha düşük değer verdi. Bu, tüm modeller veya kullanım alanları için genel bir kazanan sonucu değildir. [Tam aggregate JSON](../reports/multimodal/vidore-laya-industrial-v1/summary.json), 126 None ve 378 Laya koşulunun 17 metriğini, eşleştirilmiş farkları ve kapsam bilgilerini tam hassasiyetle içerir. Havuzlanmış koleksiyon doğruluğu, güven aralığı, p-değeri veya istatistiksel anlamlılık iddiası yoktur.

## Ölçülen süreler ve kapsamları

| İş | Süre (sn) | Ölçüm kapsamı |
|---|---:|---|
| None CPU türetme | 43.919921 | Saklanan sıralamalardan 126 koşulun türetilmesi; yeni retrieval çıkarımı değil. |
| Laya başarılı skor çağrıları | 4954.630108 | 283 sorgu ve 80.768 benzersiz sorgu–sayfa çiftinin ortak aday birleşimi; senkronizasyon dahil, yükleme/temizleme/doğrulama hariç. |
| Laya 18 skor alt süreci | 10622.563553 | Monoton saatle başlatma–sonlanmayı bekleme toplamı; yükleme, temizleme, doğrulama ve çalışma dahil. |
| CPU sonlandırıcı | 128.976723 | Ayrı sonlandırıcı alt sürecinin başlatma–sonlanmayı bekleme süresi. |
| J retrieval süreci | 1274.483695 | Ayrı yerel J alt sürecinin tam işlem süresi. |
| S retrieval süreci | 5516.972444 | Ayrı yerel S alt sürecinin tam işlem süresi. |

Bu satırlar aynı kapsamda uçtan uca ölçümler değildir; hız oranı hesaplanmaz. None için bütün retrieval kanallarının yeni çıkarım süresi ölçülmüş değildir. Ana koordinatörün tam monoton süresi, model yeniden yükleme ve cache temizleme süreleri ayrı ölçülmedi; bilinmeyen süreler sıfır değildir. Skor ve süreç süreleri birbirinden çıkarılarak yükleme veya temizleme süresi türetilmez. Ortak skorlar birden çok koşulda kullanıldığından süre koşul sayısıyla çarpılmaz. Cache okumak çıkarım değildir; çevrimiçi sorgu gecikmesi, koşul başına gecikme ve p95 ölçülmedi.

## Koşullar ve doğrulama

Laya mevcut yerel `convaiinnovations/laya-multilingual` modeliyle MPS float32, batch 8 ve en çok 8192 token kullanır. Korunan SDK dil argümanı `tr`dir; İngilizce sorgularla dil eşleşmesi sağlanmış bir deney değildir. B CPU; korunmuş G/E/N NVIDIA L4 bfloat16; yeni S/J yerel MPS bfloat16 koşullarıdır. Bu donanım ve sayısal hassasiyet farkları JSON’da ayrı tutulur. En çok 16 sorguluk süreçler operasyonel bir sınırdır; kapasite veya liste başına ayrı yeniden sıralamayla bit düzeyinde eşitlik kanıtı değildir.

Bağımsız bilimsel inceleme 283 sorgunun aday grafiğini, 80.768 benzersiz skor çiftini, dağıtımı, sıralama/birleştirmeyi, 106.974 sorgu–koşul sonucunu ve 17 metriği doğruladı. Laya metriklerinde en büyük mutlak fark 1,11×10⁻¹⁶ oldu. Önceden bağımsız kabul edilmiş 126 None koşulu, sekiz eski koşulla uyum ve S/J incelemeleri yeniden hesaplanmadan devralındı. Gerçek 18 skor alt süreci, bir sonlandırıcı ve ana koordinatör başarıyla kapandı. Ana araç sinyal alanı dönmediği için sinyal sonucu bilinmiyor; başarılı alt süreçlerin kaydedilmiş sinyal değeri ayrıdır.

Önceki tokenizer aşamasında alt süreç başarılı, denetleyici başarısız kapanmıştı. Ayrı bir doğrulama süreci mevcut çıktıyı tokenizer/model çalıştırmadan kabul etti; ilk başarısız kayıt korunur. Eski sonuçlar, incelemeler ve hata kayıtları değiştirilmedi. Güncel dosya bütünlüğü geçmiş embedding üretiminin geriye dönük tasdiki değildir. Yayında yalnız toplu ölçümler bulunur; sorgu/metin, sayfa kimliği, özel dosya yolu ve ham kanıt dökümü bulunmaz.
