# FLEURS Türkçe: 25 sorguluk paraphrase pilotu için başarı ve zaman protokolü

22 Ekim 2026. Bu belge, mevcut kayıtlardaki süreleri yorumlar ve henüz çalıştırılmamış 25 sorguluk pilotun ölçüm sözleşmesini tanımlar. Yeni sorgu, model çıkarımı, başarı skoru veya gecikme sonucu üretildiği anlamına gelmez. Eski literal sorguların sonuçları korunur; [sonraki aşama planındaki](fleurs-turkish-next-stage.tr.md) anlam ve alaka kontrolü koşulları geçerlidir.

[Mevcut multimodal planda](multimodal-plan.tr.md) p50/p95 ile generation, embedding, indeks, rerank, bellek ve disk kullanımının ayrı ölçülmesi zaten isteniyordu. Uygulamanın tarihsel zaman kayıtları bu hedefin yalnız bir bölümünü karşılıyor; yeni 25 sorgu için henüz ölçüm yapılmadı.

## Mevcut ölçümler: 743 belge, 329 literal sorgu

Mevcut galeri 743 ses kaydıdır. 329 sorgu, normalize edilmiş referans transkript gruplarıdır; yeniden ifade edilmiş anlamsal sorgular değildir. B/G/E belgeleri aynı Whisper metnini, N gerçek sesi, J ses ile aynı Whisper metnini kullanır. Bu koşuda N/J sorguları da metindir. Laya, N'nin bulduğu kayıtlar dahil, adayların Whisper metnini skorlar.

[Fresh arama raporundaki](../reports/multimodal/fleurs-tr_tr-whisper-all-retrieval/report.json) toplamlar aşağıdadır. Her neural kanal için belge `new_rows=743`, sorgu `new_rows=329`, cache satırı sıfır ve `ranking_cache_hit=false` kayıtlıdır. Tablodaki süreler saniyedir; yuvarlanmamış değerler kaynak raporda korunur.

| Kanal | 743 belge encode | 329 sorgu encode | 329 exact-search toplamı | Kayıtlı batch |
|---|---:|---:|---:|---:|
| G — BGE-M3 metin | 7,239671 | 1,204811 | 0,022302 | 8 |
| E — EG2 metin | 4,554839 | 1,473654 | 0,021378 | 8 |
| N — EG2 doğrudan ses | 74,520858 | 9,254250 | 0,034805 | 1 |
| J — EG2 ses + metin | 94,355463 | 7,928325 | 0,023427 | 1 |

Bu toplamların her biri farklı bir aşamadır. Encode toplamını pipeline toplamı olarak adlandırmak veya 329'a bölerek p50/p95 elde etmek doğru değildir. B için kayıtlı **0,438832 saniye**, BM25 indeksinin kurulması ile 329 sorgunun aramasını birlikte kapsar; yalnız online arama süresi değildir.

| Diğer tarihsel ölçüm | Değer | Ölçülen kapsam |
|---|---:|---|
| Whisper generation | 816,213447 sn | 743 fresh transkript; senkronize model generation; feature hazırlığı hariç |
| Laya fresh skorlamaları | 652,524489 sn | Matrisin 15 hücresinde toplam 83.922 yeni query–candidate çifti |
| Laya cache kullanımları | 3.138.431 | Tekil çift sayısı değil, tekrar kullanılan skor çifti oluşumları; fresh süreye dahil değil |

Whisper süresi [ASR özetindeki](../reports/multimodal/speech-asr-summary.json) tarihsel generation kaydıdır; tam ASR hazırlığı, dosya I/O'su veya bütün retrieval pipeline'ının wall-time'ı değildir. ASR 743 kayıt için bir kez yapılmıştır. Her B/G/E/J veya Laya varyantına yeniden 816 saniyelik gerçek çalışma yazılmaz.

[Laya raporundaki](../reports/multimodal/fleurs-tr_tr-whisper-text-rerankers/report.json) `latency_scope`, `fresh_missing_pairs_only; reused scores excluded` değerindedir. `total` bütçeli 31 K50 hücresinin tümü cache kullanır; yeni çift sayısı sıfır ve fresh süre/yüzdelikler `null` olur. Bu değerler sıfır gecikme anlamına gelmez. `per_channel` K50 hücrelerindeki süreler de tam aday listesine ait değildir: örneğin B+Laya için 9.470 fresh ve 6.549 cached çift ile p50/p95 yaklaşık 248/392 ms kaydedilmiştir. Bu, 50 adayın tamamını yeniden skorlayan bir sorgunun p50/p95'i değildir. Yeni pilotun tam K50 gecikmesi bu sayılardan tahmin edilmeyecektir.

## Cihaz ve zamanlayıcı sınırları

Raporlanan yapılandırma G/E/N/J için MPS ve bfloat16'dır. G/E batch 8, N/J batch 1'dir. Whisper MPS/float32, Laya SDK MPS/float32 ve batch 8 kullanır. Dolayısıyla bütün süreler aynı dtype/batch koşulunda ölçülmüş gibi karşılaştırılmaz. Model revision'ları raporlarda sabittir; yalnız model ailesinin aynı olması aynı execution path anlamına gelmez. MPS etiketi tek başına fiziksel Apple çipi, bellek durumu, güç/termal durum ve eşzamanlı yükün bütün ölçümlerde aynı olduğunu kanıtlamaz.

E/N/J'nin eski sorgu encode toplamlarından çıkarılan ortalamalar, farklı batch ve olası lazy-load etkileri nedeniyle model temelli hız üstünlüğü kanıtlamaz.

[Mevcut runner](../src/rag_benchmark/multimodal.py) encode süresini `adapter.encode` çağrısının çevresinde ölçer. Adapter'lar modeli lazy yükleyebildiği için ilk çağrı model yüklemeyi de içerebilir. Girdi hazırlığı ve doğrulamanın hangi bölümlerinin içeride kaldığı adapter'a bağlıdır. Bunlar saf forward-pass süreleri veya doğrulanmış warm-serving süreleri olarak etiketlenmez.

[Native/joint adapter](../src/rag_benchmark/multimodal_models.py) ve [Laya SDK yolu](../src/rag_benchmark/models.py) dönüşten önce cihaz senkronizasyonu içerir; metin encoder yolu NumPy/CPU çıktısına dönüşür. Bununla birlikte mevcut ortak dış zamanlayıcı her ölçümün öncesinde ve sonrasında aynı açık GPU bariyerini uygulamaz. Ortak, ayrıştırılmış cold-start/warm-up sözleşmesi de kayıtta yoktur. Fusion, cache I/O ve uçtan uca sorgu wall-time'ı ayrı ölçülmemiştir. Mevcut rapor, yeni pilotun başarı–servis gecikmesi karşılaştırmasını tek başına karşılamaz.

## Yeni 25 sorguluk pilotun sabit kapsamı

Model skorlarına bakılmadan seçilen 25 farklı kaynak grubunda, anlamı koruyan birer Türkçe paraphrase sorgusu hazırlanır. Aynı 743 kayıt bütün yöntemlerin galerisidir. Etiketler ve sorgu politikası koşudan önce sabitlenir; anlam/alaka kontrolü tamamlanmamış örnekler doğrulanmış semantik benchmark olarak sunulmaz. Sorgu politikasının geliştirildiği bu küçük pilot, bağımsız nihai test başarısı sayılmaz.

Yeni 25 sorgunun girdisi metindir. N'nin ham ses kullanması aday temsilini tarif eder; bu pilot sesli sorgu deneyi değildir.

Beş tek kanal ve önceden seçilip manifestte sabitlenen fusion koşulları karşılaştırılır. Fusion üyeleri, kaynak derinliği, RRF k=60, aday bütçesi, tie-break ve metrik kesimleri başarı sonuçlarına bakılarak değiştirilmez. İlk zaman pilotunda `channel_budget=100` ve `budget_mode="total"` kullanılır: fusion girişinin toplam retrieval bütçesi 100'dür. RRF k=60 ile fusion sonrasında `candidate_k=50` kesimi yapılır; Laya'ya en fazla 50 aday gider. `per_channel` modunda kanal başına 100 retrieval adayı ayrı kalite ablation koşuludur. Bu sabit bütçeli zaman pilotu, sonraki aşama planındaki önce beş tek kanal, ardından 31 subset × iki bütçe kalite karşılaştırmasını iptal etmez.

None/Laya çifti aynı fused ranking ve en fazla 50 adaylık ön listeden türetilir. Laya bu listenin tamamını yeniden sıralar; None'ın top-20 ve nDCG@10 dahil metrikleri aynı ön listenin ilgili kesimlerinden hesaplanır. İki koşula da bu 50 adaylık desteğin dışından sonuç eklenmez. BM25 kısa liste döndürürse gerçek sayı korunur; sıfır puanlı aday eklenmez.

Birincil başarı metriği Hit@5; Hit@1/20, çok-pozitif Recall@5/20, MRR@10 ve nDCG@10 destekleyicidir. Koşulların ortak 25 sorgusu, tamamlanan/başarısız/etiketi çözülmemiş sayıları açık raporlanır. Hatalı sorgular sessizce çıkarılarak başarı veya gecikme iyileştirilmez. Bu retrieval deneyinde cevap üretimi veya QA doğruluğu ölçülmez.

## Ölçülecek aşamalar

| Aşama | Yeni ölçüm sözleşmesi |
|---|---|
| Offline hazırlık | ASR, belge encode, indeks kurma ve yeniden kullanım doğrulaması ayrı. Mevcut 743 ASR ve belge vektörleri korunur; eski işi tekrar yapmak bu planın parçası değildir. Yeni çalışmada gerçekten yapılan artifact yükleme/doğrulama wall-time'ı ayrıca tutulur. |
| Cold başlangıç | Model yükleme, indeks/vektör açma ve ilk çağrı/derleme etkileri ayrı zamanlanır. Offline ile aynı işin iki kez toplanmaması için aşama sınırları manifestte yazılır. |
| Warm sorgu | Resident model ve önceden tanımlı warm-up sonrası fresh query encode, search, fusion ve sonuç sıralama ayrı; aynı isteğin başlangıcından sonuç hazır olana kadar kapsayıcı wall-time da ölçülür. |
| Warm sorgu + Laya | Aynı sorgunun retrieval sürecine ek olarak gerçek K50 aday kümesinin tamamı fresh skorlanır; token hazırlığı, batch inference ve nihai sıralama kapsamı kaydedilir. Rerank-only ve kapsayıcı toplam ayrı tutulur. |
| Cache-hit servis | Ölçülecekse kendi satırıdır. Vektör/skor/result cache okuması model inference diye yazılmaz; fresh ölçümlerin yerine geçmez. |

Online karşılaştırmanın temel birimi tek sorgudur: concurrency 1, resident model ve sabit execution topology. Query batch 1 kullanılır; belge hazırlığının ve Laya aday çiftlerinin batch değerleri ayrıca kayıtlıdır. Toplu 25-query throughput istenirse ayrı deney ve ayrı birim olarak raporlanır. Bellek nedeniyle bütün modeller aynı anda resident olamıyorsa model değiştirme maliyeti gizlenmez; gerçek sequential/parallel çalışma düzeni ve yükleme sınırı açık yazılır.

Yeni zamanlayıcı monoton saat kullanır. GPU işi içeren her ölçümde aynı MPS cihazı başlangıçtan hemen önce ve bitişten önce senkronize edilir. Warm-up sayısı/girdisi önceden belirlenir, başarı örneklemine karıştırılmaz ve maliyeti ayrıca kaydedilir. Gerçek cihaz, backend, dtype, batch, concurrency, model revision, truncation ve cache durumu raporlanır; fallback veya retry varsa sayısı ve süre etkisi saklanmaz.

Bellek ve disk maliyeti gecikmeden ayrı alanlardır. Kaydedilebilen peak/allocated/resident bellek ölçülerinin türü ve kapsamı, model/cache/indeks disk boyutları belirtilir; ölçülmeyen alanlar sıfırla doldurulmaz.

Laya'nın tam K50 ölçümünde, aynı sorgu ve koşulun bütün gerçek adayları modelden fresh skor almalıdır. Önceki K20, başka kanal veya önceki sorgu çalışmasının puanları bu zaman ölçümünün bir bölümünü atlayamaz. Yeni pilot için ayrı bir ölçüm kimliği kullanılır; eski cache silinmez veya değiştirilmez. Sonuçların kalite değerlendirmesinde paylaşılması, paylaşılan skorların yeniden fresh latency kazanması anlamına gelmez. Gerçek aday sayısı 50'den azsa bu sayı ve tamamı skorlanan çift sayısı birlikte raporlanır.

Her koşulda 25 sorgunun bireysel kapsayıcı sürelerinden p50/p95 hesaplanır; örnek sayısı ve percentile yöntemi yazılır. Aşama p95'leri veya medyanları toplanarak uçtan uca p95/medyan üretilmez. Süre tekrarları yapılırsa tekrarlar ve sorgular ayrı sayılır; aynı 25 sorguyu tekrar etmek bağımsız başarı örneklemini büyütmez. 25 örnekli p95 küçük pilotun kuyruk tahminidir; üretim SLA'sı veya kararlı nüfus yüzdeliği değildir.

## Paylaşılan iş ve encode yeniden kullanımı

ASR metni B/G/E/J ile metin reranker'ları arasında paylaşılır. N-only yolunun kaynak ses encode işi ASR'ye bağlı değildir; N+Laya ise aday metni için ortak ASR'den yararlanır. Tekil workflow maliyeti ile bir yöntemin bağımsız kurulacağı senaryonun maliyeti ayrı gösterilir; ortak ASR veya indeks kurma işi kanal sayısıyla çarpılmaz.

Yeni sorguların eski query/ranking/rerank cache'lerini kullanması yasaktır. Belge vektörlerinin aktarımı ancak corpus sırası/byte hashleri, model revision, belge prompt'u, preprocessing, boyut, dtype ve normalizasyon eşitliği doğrulanınca yapılır. Mevcut dataset cache kimliğine yeni sorguları eski sorgularmış gibi bağlamak yerine açık bir yeniden kullanım kaydı gerekir.

Aynı yeni sorgu için query encode paylaşımı, yalnız etkili girdi ve task prompt'u, model/weight revision, tokenizer/processor, preprocessing, boyut/normalizasyon, dtype/backend ve yürütme yolu eşitliği kanıtlanırsa mümkündür. E/N/J'nin aynı model adını taşıması yeterli değildir: text-only yükleme ile native/joint yolunun eşdeğerliği varsayılmaz. G vektörü başka encoder'ın belge uzayında kullanılamaz.

Koşullar arasında paylaşım varsa gerçek tek encode çağrısı ve tüketen koşullar kaydedilir. Kanal veya varyant sayısı kadar mükerrer süre faturası kesilmez. Her yöntemin bağımsız online gecikmesi ayrı fresh ölçümle elde edilir; paylaşılan çalışma grafiğinin toplam süresi ise gerçek başlangıç/bitişten ölçülür. Önceki sürelerden yeniden kurulan karşı-olgusal toplam varsa açıkça hesaplanmış tahmin diye ayrılır. Paralel topology'de aşama sürelerinin toplamı wall-time yerine geçmez.

## Başarı–zaman tablosu ve amortizasyon: yalnız plan

Aşağıdaki tablo sonuç şablonudur; çizgiler ölçülmemiş alanlardır. Fusion koşulları manifestteki sabit üyeleriyle ayrıca satır alır. Laya ile None aynı aday kümesine bağlı eşleştirilmiş karşılaştırmadır.

| Planlanan koşul | Hit@5 | Recall@5 | MRR@10 | nDCG@10 | Warm toplam p50/p95 | Offline maliyet |
|---|---:|---:|---:|---:|---:|---:|
| B / B + Laya K50 | — | — | — | — | — | — |
| G / G + Laya K50 | — | — | — | — | — | — |
| E / E + Laya K50 | — | — | — | — | — | — |
| N / N + Laya K50 | — | — | — | — | — | — |
| J / J + Laya K50 | — | — | — | — | — | — |
| Sabit fusion / aynı fusion + Laya K50 | — | — | — | — | — | — |

Yayımlanan sonuçta her slash ile gösterilen çift iki ayrı satır olur. Pareto incelemesi önceden seçilen başarı metriği Hit@5 ile warm kapsayıcı p50 için, ayrıca p95 için yapılır. Bir koşul diğerinden en az aynı başarıda ve en fazla aynı gecikmedeyse, bir eksende de sıkı üstünlük varsa gözlenen pilotta baskın sayılır. Offline maliyet ve diğer başarı metrikleri saklanmaz; tek keyfî birleşik puana indirgenmez. Küçük farklar istatistiksel veya genel üstünlük diye sunulmaz.

Seri ve sabit topology için planlanan maliyet hesabı:

`T_total(Q) = T_offline + Q × mean(T_online)`

Burada Q gelecekteki sorgu sayısı, T_offline o senaryo için tekil hazırlık maliyeti, T_online gerçek fresh sorgunun kapsayıcı süresidir. Bu formül ölçülmüş gelecek sonuç değildir. Cold başlangıç offline kapsamına alınmamışsa ayrıca `N_start × T_cold` eklenir; aynı yükleme iki kez sayılmaz. Sorgu başına amortize hazırlık `T_offline / Q` olur. Q ile çarpılacak değer p50 veya p95 değil ortalamadır; gerçek sonlu koşuda en doğru toplam bireysel sürelerin toplamıdır. Paralellik, dinamik batch veya kaynak çekişmesi varsa bu seri formül yerine ölçülmüş gerçek topology wall-time'ı kullanılır.

## Yayın sınırı

Public raporda yalnız aggregate başarı/süreler, örnek ve çift sayıları, cache/fresh ayrımı, cihaz/yapılandırma, sabit model ve kaynak SHA'ları yayımlanır. Ham sorgu, transkript, ses içeriği, yerel mutlak yol, özel kimlik listesi veya hata metni eklenmez. Eski 329 literal sorgunun başarısı ile yeni 25 paraphrase pilotunun başarısı aynı paydada toplanmaz. Bu belgenin eklenmesi, planlanan ölçüm veya model doğrulamasının yapıldığı anlamına gelmez.
