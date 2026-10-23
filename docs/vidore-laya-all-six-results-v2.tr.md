# ViDoRe: altı koleksiyonda Laya karşılaştırması

Altı PDF koleksiyonunun **12.648 sayfa ve 1.921 sorgusu** için hesaplama ve tam bağımsız bilimsel çıktı incelemesi tamamlandı. **756 Laya’sız (None) koşul** korunarak **2.268 Laya koşuluyla** karşılaştırıldı. Industrial koleksiyonu bu altı koleksiyona dahil değildir; onun yeni testleri ayrı yürütülüyor. Geniş tüm-modalite test matrisi tamamlanmış sayılmaz.

B = BM25, G = BGE-M3 metin, E = EG2 metin, N = EG2 ham sayfa görseli, S = ColQwen2.5, J = EG2 ortak metin + görsel. Sorgular yazılı metindir. Altı kanalın 63 boş olmayan alt kümesi, `per_channel` ve `total` bütçeleriyle kullanılır. Bütçe 100, RRF sabiti 60; Laya aday kesimleri K=20/50/100’dür. `per_channel` her kanaldan 100, `total` birleşimden önce kanallar arasında paylaştırılmış toplam 100 aday alır.

Hit@K ilk K içinde en az bir ilgili sayfa bulunma oranıdır. Recall@K ilgili sayfaların ne kadarının ilk K içinde bulunduğudur. Aşağıdaki oranlar yüzde, farklar yüzde puandır; JSON tüm 17 metriği tam hassasiyetle içerir.

## Sabit örnek karşılaştırma: tüm altı kanal, per_channel, Laya K=50

Bu tablo önceden belirtilen ana bütçe/K ayarında B+G+E+N+S+J alt kümesini gösterir; en iyi koşullar seçilerek oluşturulmamıştır. None tüm elde edilen birleşimi sıralar, Laya ilk 50 adayı yeniden sıralar. Dolayısıyla fark hem aday kesimini hem yeniden sıralamayı içerir; yalnız reranker etkisi değildir.

| Koleksiyon | Sorgu | Yöntem | Hit@5 | Hit@20 | Recall@5 | Recall@20 |
|---|---:|---|---:|---:|---:|---:|
| Enerji (FR) | 308 | None | 87.99% | 96.75% | 61.46% | 82.07% |
| Enerji (FR) | 308 | Laya K=50 | 52.60% | 83.44% | 29.46% | 62.56% |
| Enerji (FR) | 308 | Laya − None (yp) | -35.39 | -13.31 | -31.99 | -19.51 |
| Finans (EN) | 309 | None | 88.67% | 99.03% | 56.36% | 78.96% |
| Finans (EN) | 309 | Laya K=50 | 47.90% | 87.06% | 21.17% | 53.71% |
| Finans (EN) | 309 | Laya − None (yp) | -40.78 | -11.97 | -35.19 | -25.25 |
| Finans (FR) | 320 | None | 64.06% | 85.31% | 37.22% | 61.77% |
| Finans (FR) | 320 | Laya K=50 | 34.38% | 73.44% | 13.23% | 43.60% |
| Finans (FR) | 320 | Laya − None (yp) | -29.69 | -11.88 | -24.00 | -18.18 |
| İnsan kaynakları (EN) | 318 | None | 84.91% | 95.28% | 51.03% | 75.41% |
| İnsan kaynakları (EN) | 318 | Laya K=50 | 48.74% | 84.91% | 18.42% | 52.64% |
| İnsan kaynakları (EN) | 318 | Laya − None (yp) | -36.16 | -10.38 | -32.61 | -22.77 |
| İlaç (EN) | 364 | None | 88.74% | 98.08% | 57.48% | 78.94% |
| İlaç (EN) | 364 | Laya K=50 | 60.44% | 92.31% | 30.97% | 65.73% |
| İlaç (EN) | 364 | Laya − None (yp) | -28.30 | -5.77 | -26.51 | -13.20 |
| Fizik (FR) | 302 | None | 78.15% | 93.05% | 35.72% | 62.74% |
| Fizik (FR) | 302 | Laya K=50 | 51.66% | 86.42% | 18.41% | 49.30% |
| Fizik (FR) | 302 | Laya − None (yp) | -26.49 | -6.62 | -17.32 | -13.44 |

Tüm 63 alt küme, her iki bütçe ve üçK için [tam aggregate JSON](../reports/multimodal/vidore-laya-all-six-v1/summary.json) mevcuttur. Her Laya hücresinde aynı koleksiyon, sıralı sorgu grubu, alt küme ve bütçedeki None ile eşleştirilmiş 17 metrik farkı vardır. Koleksiyonlar arasında havuzlanmış doğruluk, güven aralığı, p-değeri veya genel kazanan iddiası yapılmaz.

## Ölçülen Laya skor süreleri

Bu süreler başarılı ortak-aday-birleşimi skor çağrılarının toplamıdır; çağrı içindeki skorların elde edilmesi ve senkronizasyonu dahildir. Model yükleme, süreçler arasında yeniden yükleme, cache temizleme, doğrulama ve sonucu bilinmeyen tarihsel denemeler hariçtir. Aynı kayıtlı skor birden çok koşula dağıtıldığından süre koşul sayısıyla çarpılmaz; cache okumak çıkarım değildir. V4/V2/V3, ayrı işlem nesillerini belirtir.

| Koleksiyon | İşlem nesli | Başarılı sorgu | Benzersiz sorgu–sayfa çifti | Skor çağrıları toplamı (sn) |
|---|---|---:|---:|---:|
| Enerji (FR) | v4 | 229 | 59432 | 7468.234290 |
| Enerji (FR) | v2 | 79 | 20106 | 2497.901177 |
| Finans (EN) | v2 | 231 | 58960 | 9279.962571 |
| Finans (EN) | v3 | 78 | 19725 | 2816.855293 |
| Finans (FR) | v3 | 320 | 84227 | 12376.808088 |
| İnsan kaynakları (EN) | v3 | 318 | 77952 | 11004.656506 |
| İlaç (EN) | v3 | 364 | 102067 | 5134.445210 |
| Fizik (FR) | v3 | 302 | 74534 | 2532.326273 |

## Sürelerin karşılaştırılabilir kapsamı

Laya’sız eski embedding ve bütün retrieval çıkarımının aynı kapsamda ölçülmüş süreleri bu yayında yoktur; bilinmeyen değerler sıfır yapılmaz. Finans FR ve İK için daha önce yayımlanan cache tamamlama süreleri aşağıdadır. Bunlar doğrulama, CPU exact search, matris türetme ve yazma işidir; Laya skor süresiyle hız oranı hesaplanamaz.

| Laya’sız CPU tamamlama | İş akışı (sn) | Eksik kanal exact search (sn) | Yeni embedding |
|---|---:|---:|---|
| Finans (FR) | 10.845186 | 0.050651 | yapılmadı |
| İnsan kaynakları (EN) | 8.256723 | 0.029692 | yapılmadı |

Gerçek başarılı alt süreçlerin başlangıç–bekleme UTC zaman farkları [ayrı süre JSON’unda](../reports/multimodal/vidore-laya-all-six-v1/timing-scopes.json) bulunur. Bu gözlemler süreç yükleme, hesaplama ve diğer işleri içerir; monoton saat ölçümü veya çevrimiçi sorgu gecikmesi değildir. Skor toplamından çıkarılarak model yükleme/temizleme süresi türetilmez. Finansın eski 231 sorgusunun224’ü tamamlanmış 14 alt süreçten,7’si terminal süreç sonucu bilinmeyen kesintili bölümden gelir; bu 7 sorguya başarılı süreç süresi uydurulmamıştır.

## Koşullar ve bağımsız doğrulama

Laya mevcut yerel `convaiinnovations/laya-multilingual` modelinin `1720e3e3357cfe1e281542e223f8273b0890ca34` sürümüyle MPS float32, batch 8 ve en çok 8192 token kullanır. Korunmuş SDK dil argümanı `tr`dir; İngilizce/Fransızca verilerle dil eşleşmesi sağlanmış deney değildir. Eski retrieval CPU/L4/device/dtype koşulları JSON’da kanal bazında ayrıdır. V2/V3 kurtarma süreçlerinde her sorgu sonrası senkronizasyon ve cache temizliği ile 64 GiB süreç RSS/MPS driver sınırları uygulanmıştır. Bu süreçlerdeki en çok 16 sorgu sınırı operasyonel bir tercihtir; kanıtlanmış kapasite sınırı değildir. Devralınan 229 V4 sorgusuna bu cache temizliği ve 16 sorguluk süreç düzeni atfedilmez; V4 aynı 64 GiB korumaları kullanmıştır.

Bağımsız inceleyici üretim metrik/dağıtım fonksiyonlarını çağırmadan tüm 1.921 sorgunun aday grafiğini, 497.003 benzersiz skor çiftini, 726.138 sorgu–koşul sonucunu, 17 metriği, sayaçları, süre kapsamını ve kapanış zincirini doğruladı. Metriklerde en büyük mutlak fark 2,22×10⁻¹⁶ oldu. Yeni 87 skor alt süreci ve 5 CPU sonlandırıcı ile geçmiş 20 gerçek başarılı bekleme kaydı incelendi. None metrikleri, önceden kabul edilmiş tokenizer/girdi ve 229+231 yeniden kullanım incelemeleri tekrar çalıştırılmadı. Yeni model çağrısıyla sonuçlar yeniden üretilmedi.

Eski V4 başarısız çıkışı, reboot sırasında terminal sonucu bilinmeyen V2 ana/aktif alt süreç ve çıkarım sayısı bilinmeyen iki tarihsel sorgu denemesi ayrı korundu. Yeni seriden önceki kurulum hatasında hiçbir worker/model/sorgu başlamadı; başarılı ikinci başlatma ayrı kaydedildi. Başarılı yeni ana süreç gerçek exit 0 ile kapandı; parent tool sinyal alanı dönmediği için sinyal sonucu gözlenmiş sayılmaz. Güncel dosya bütünlüğü, eski embedding üretiminin veya geçmiş fiziksel diskin geriye dönük tasdiki değildir.
