# Türkçe konuşma: mevcut testin değerlendirmesi ve sonraki aşama

22 Ekim 2026. Durum: mevcut yayımlanmış sonuçlar üzerinden yöntem incelemesi ve yeni deney protokolü. Bu belge yeni benchmark sonucu, model çalıştırma kaydı veya tamamlanmış insan etiketlemesi değildir.

## 1. Mevcut test neyi ölçüyor?

FLEURS Türkçe test görünümünde 743 ses kaydı, normalize edilmiş transkriptlerine göre 329 sorgu/pozitif grup var. Sorgu, kayıtta söylenen cümlenin referans metni. Aynı cümleyi söyleyen birden fazla kayıt birlikte ilgili sayılıyor. Bu, metinle konuşma kaydı bulma görevi; açık uçlu soru cevap veya farklı ifadelerle anlam arama testi değil. FLEURS'ün resmî verisinden türetilmiş proje retrieval protokolü, resmî bir FLEURS semantik arama liderlik tablosu olarak sunulmaz.

Beş kanal mevcut:

| Kanal | Aday girdisi |
|---|---|
| B — BM25 | Kaynak sesten Whisper'ın çıkardığı metin |
| G — BGE-M3 | B ile aynı Whisper metni |
| E — EG2 metin | B ile aynı Whisper metni |
| N — EG2 doğrudan ses | Gerçek ses dalgası |
| J — EG2 birlikte | Gerçek ses + aynı Whisper metni |

Referans transkript aday ASR metni olarak kullanılmıyor. Whisper yalnız kaynak sesi görüyor. Referanslar sorgu/pozitif grup ve sonradan WER hesabı için kullanılıyor. B/G/E aynı metni paylaşırken N hiçbir aday transkriptini tüketmiyor. Laya ve BGE-reranker, N ile bulunan adaylar dahil, yalnız Whisper metnini yeniden sıralıyor.

## 2. Korunan tamamlanmış sonuçlar

| Ölçüm | Yayımlanmış sonuç |
|---|---:|
| Whisper kapsamı | 743/743 kayıt; boş çıktı yok |
| Whisper kelime hata oranı (WER) | %7,2924; 965 düzenleme / 13.233 referans kelime |
| EG2 doğrudan ses Hit@1 | %99,70 |
| EG2 doğrudan ses Hit@5 | %100 |
| EG2 doğrudan ses çok-pozitif Recall@5 | %99,54 |
| B/G/E/J tek kanal Hit@1 ve Hit@5 | %100 |
| Arama kapsamı | 31 boş olmayan kanal birleşimi × 2 aday bütçesi = 62 koşul |
| Arama + metin reranker kapsamı | 62 temel + 186 Laya + 186 BGE = 434 koşul |

İki bağımsız metrik uygulamasının doğruladığı eski sonuçlar korunur; bu inceleme eski sıralamaları veya model çıkarımını tekrar çalıştırmaz. Ters yöndeki ses→referans transkript araması da ayrı tamamlanmış görevdir. O galeri açıkça referans metin içerir; ASR üretimi değildir.

Sonuçlar temel eşleştirmenin çalıştığını gösteriyor. Sorguların birebir cümle olması ve 743 kayıtlık galeri birçok yöntemi tavana taşıyor; %100 Hit@5 genel Türkçe konuşma anlamayı kanıtlamaz. WER'den anlamsal doğruluk çıkarılmaz. FLEURS okunmuş konuşmaya odaklanır; doğal sohbet ve gürültülü kullanımın başarısı bu sonuçlardan çıkarılamaz. Aynı transkript grubunun ötesindeki konuşmacı/konu bağımlılıkları ve kamu verisinin eğitimde görülmüş olma olasılığı bu deneyle giderilmiş değildir.

## 3. Sonraki öncelik: farklı ifadeyle aynı içeriği bulma

Yeni, ayrı isimli `FLEURS-TR paraphrase retrieval` görünümü hazırlanır. Önce model başarısına bakmadan seçilen 20–30 kaynak grubunda sorgu yazım ve alaka kontrolü yönteminin uygulanabilirliği incelenir; bu küçük dil kontrolü tam benchmark sonucu değildir. Tam kapsam hedefi, 329 kaynak cümle grubunun her biri için bir anlamı koruyan Türkçe yeniden ifade edilmiş sorgudur. Bu bir hedef sayısıdır; henüz hazırlanmış veya doğrulanmış sorgu sayısı değildir. Küçük örnek sorgu politikasını geliştirmek için kullanılırsa ilgili gruplar ayrı geliştirme kapsamı olarak etiketlenir ve kalan değerlendirme kapsamının gerçek sayısı ayrıca raporlanır.

- Sorgu yazarı kaynak cümlenin anlamını kullanabilir; hiçbir aday yöntemin sıralamasını, puanını veya hangi yöntem kazansın beklentisini görmez.
- Yeniden ifade; kişi, sayı, yer, olumsuzluk, neden-sonuç ve zaman ilişkilerini korur. Yeni bilgi veya kaynakta bulunmayan cevap eklenmez.
- Hedef düşük kelime örtüşmesi uğruna anlamı bozmak değildir. Kelime örtüşmesi sonradan betimleyici olarak raporlanır; model başarısına göre sorgu seçilmez.
- Planın insan anlam kontrolü gereği korunur. Otomatik/ajan incelemesi insan onayı olarak adlandırılmaz. Kontrol edilmemiş sorgular yalnız taslak/keşifsel statüdedir; doğrulanmış semantik benchmark diye yayımlanmaz. İnceleme için kullanıcıdan bu turda bir karar talep edilmiyor.
- Dar ve tam anlamı koruyan yeniden ifadelerde eski pozitif grup başlangıç etiketi olabilir; başka kayıtların da ilgili hale gelip gelmediği kontrol edilir. Daha genel konu/niyet sorguları ayrı deneydir ve yeni alaka etiketleri gerektirir. Etiketlenmemiş bütün kayıtlar otomatik olarak yanlış ilan edilmez. Alaka etiketleri koşudan önce dondurulur; çözülmemiş etiket kapsamı varsa standart tam-galeri metriği doğrulanmış sonuç diye yayımlanmaz.
- Yeni query ID, dataset revision ve sorgu/etiket hashleri kullanılır. Eski literal sorgular ve raporlar değiştirilmez; iki görünüm aynı başarı paydasına karıştırılmaz.
- Prompt/model/fusion/K seçimleri değerlendirme sorgularının sonuçlarına bakılarak ayarlanmaz. Yeni ayar gerekirse ayrı geliştirme kaynağı veya önceden ayrılmış kaynak-cümle grupları kullanılır. Mevcut 329 grup daha önce incelenmiş olduğundan yeni ifade görünümü bütünüyle hiç görülmemiş bir veri kümesi olarak sunulmaz.

Yalnız açıklama amaçlı, veri kümesinden alınmamış örnek:

> Kayıt: “Yoğun kar yağışı yüzünden uçuşlar ertelendi.”
>
> Yeni sorgu: “Şiddetli karın uçak seferlerini geciktirdiğinin anlatıldığı kayıt.”

## 4. Aynı koşullarla karşılaştırma

Galeri aynı 743 kayıt olarak korunur. B/G/E aynı mevcut Whisper metnini, N aynı sesi, J aynı ses+Whisper metnini kullanır. Önce geliştirme pilotunda beş tek kanal ve tek sabit bütçeyle yeni sorgu/cache bağlarının işlediği doğrulanır; bu pilot tam sonuç değildir. İlk tam arama dalgası 31 kanal birleşiminin iki bütçedeki yeniden sıralamasız sonuçlarıdır; ardından yeni sorguların aynı adayları üzerinde Laya ve BGE-reranker karşılaştırılır. K=20/50/100, RRF k=60 ve mevcut iki aday-bütçe tanımı ayrı tutulur. Gemma'nın eski geniş plandaki bekleyen ses reranker işi ayrı kalır; bu semantik görünümün sonucu diye sayılmaz. Jev/Clef eklenmez.

Birincil metrik Hit@5; Hit@1/20, çok-pozitif Recall@5/20, MRR@10 ve nDCG@10 da ayrı raporlanır. Hit bir doğru kayıt bulmayı, Recall ilgili kayıtların tamamını bulmayı ölçer. Varsa bir kaynak cümleden türetilen bütün sorgular aynı istatistiksel grupta tutulur. Eşleştirilmiş farklar, kazanılan/kaybedilen gruplar ve belirsizlik aralıkları raporlanırken konuşmacı/konu bağımlılığı ile çoklu karşılaştırma sınırı belirtilir.

Arama gecikmesi, sorgu encoding süresi, bir defalık ASR/aday encoding maliyeti ve reranker maliyeti ayrılır. Cache okuma hızı yeni model çıkarım hızı gibi sunulmaz. Bu ilk semantik retrieval dalgası cevap üreten bir model gerektirmez.

Mevcut süre kayıtlarının kapsamı ve yeni pilotun cold başlangıç, warm sorgu, tam Laya aday listesi, p50/p95 ve başarı–zaman karşılaştırması [ayrı ölçüm protokolünde](fleurs-paraphrase-timing-protocol.tr.md) tanımlıdır. Bu protokol bir hız sonucu değildir.

## 5. Hazır işi güvenli yeniden kullanma

Sesler, Whisper çıktıları, BM25 indeksi ve doğrulanmış G/E/N/J aday vektörleri korunur. Yeni sorgular için yeni sorgu vektörleri, sıralamalar ve gerektiğinde rerank puanları gerekir.

Mevcut `encoding_cache_identity`, aday vektörleri için de bütün dataset kimliğini kullanıyor; bu kimlik `queries.jsonl` ve `qrels.jsonl` hashlerini içeriyor. Dolayısıyla yeni sorgu görünümünü mevcut runner'a doğrudan vermek adayları gereksiz yeniden encode edebilir. Eski dataset kimliğini yeni sorgulara vermek veya kontrolü kaldırmak çözüm değildir.

Çalıştırmadan önce ayrı, açık bir yeniden-kullanım bağı kurulmalı: aynı corpus ID/sırası, medya ve ASR byte/hashleri, model revision'ı, belge prompt'u, ön işleme, vektör boyutu/hassasiyeti/normalizasyonu ve vektör checksum'u doğrulanmalı. Yeni sorgular yeni kimliğe bağlanmalı; eski query/ranking/rerank cache'leri yeni sorgu sonucuna dönüşmemeli. Mevcut `PrecomputedAdapter` kullanılacaksa bu kaynak eşitlik kanıtıyla yeni dataset'e açıkça bağlanmalı. Bu belge söz konusu bağı uygulamış veya doğrulamış sayılmaz.

## 6. Bu aşamanın sonucu ve sıradaki somut işler

Mevcut literal FLEURS testi yöntem ve sonuç kapsamı bakımından incelendi: temel retrieval ve Laya/BGE kısmı tamam, semantik yeniden ifade edilmiş sorgu testi eksik. Sonraki işler ayrı sorgu görünümü ve anlam/alaka kontrolü, doğrulanmış aday-cache aktarım bağı, ardından sorgu encoding/arama ve Laya/BGE değerlendirmesidir. Gürültü, uzun konuşma, farklı konuşmacı ve sesli yeniden ifade edilmiş sorgular ayrıca tanımlanacak dayanıklılık deneyleridir; mevcut FLEURS skorları bunları tamamlamaz.

### Dayanaklar

- [Kaynak veri kartı](https://huggingface.co/datasets/google/fleurs)
- [Donmuş FLEURS planı](multimodal-plan.tr.md#türkçe-konuşma-fleurs-tr_tr)
- [ASR ve 62 arama koşulu](../reports/multimodal/speech-asr-summary.md)
- [434 koşulluk Laya/BGE sonuçları](../reports/multimodal/speech-reranker-summary.md)
- [Bağımsız reranker sonuç denetimi](../reports/multimodal/fleurs-tr_tr-whisper-text-rerankers/independent-audit.json)
- [Ters yön sonuçları](../reports/ablations/reverse/README.md)
- [Kaynak/sorgu ayrımı](../src/rag_benchmark/multimodal_data.py)
- [Dataset ve vektör cache kimlikleri](../src/rag_benchmark/multimodal.py)
