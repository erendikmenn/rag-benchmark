# Türkçe konuşma: 25 sorguluk paraphrase ve süre pilotu

22 Ekim 2026 tarihli yerel pilot tamamlandı. **743 ses kaydı üzerinde aynı 25 yeni yazılı sorgu**, beş kanalın 31 boş olmayan alt kümesi, iki retrieval bütçesi ve None/Laya karşılaştırmasıyla ölçüldü: **124 hücre, 3.100 sorgu-hücre sonucu, 1.550 eşleşmiş gerçek sorgu çalıştırması**. Her hücre aynı 743 kaydı arar; içerik sayısı koşul sayısıyla çarpılmaz.

Bağımsız inceleme 3.100 metrik sonucunu, 124 toplulaştırmayı, aday eşitliğini, gerçek başlangıç/bitiş zamanlarını ve p50/p95 hesaplarını yeniden doğruladı. Süreç gerçek shell bekleme kaydıyla **exit 0** verdi. Tüm iş akışı **696,71 saniye (yaklaşık 11 dakika 37 saniye)** sürdü. Bu süre soğuk hazırlık, doğrulama ve kayıt işlemlerini de kapsar; aşağıdaki sorgu gecikmeleri ayrı ölçümdür.

## Sonucun kapsamı

Sorgular özgün referans metinlerinin anlamını koruyan yeniden anlatımlardır; kaynak gruplarından aktarılan 54 pozitif etiket kullanılır. Her sorgunun anlamı bağımsız ajan incelemesinden geçti, ancak **insan doğrulaması ve galerinin tamamında semantik uygunluk etiketlemesi yapılmadı**. Bu keşifsel bir kaynak grubu bulma pilotudur. Yeni soruların genel başarısı, cevap üretimi, konuşulmuş sorgu, gürültü dayanıklılığı veya uzun konuşma performansı ölçülmedi.

Her iki bütçede de Laya olmadan bütün alt kümeler Hit@5/20 ve Recall@5/20'de %100'e ulaştı. Ana `total` bütçede Hit@1 de bütün 31 alt kümede %100 oldu. **25 sorguda test hâlâ kolay ve doygun**; bununla modeller arasında genel bir kazanan ilan edilemez. Laya ile Hit@5 %92–100, Recall@5 %88–96,67 aralığında; Hit@20 ve Recall@20 bütün hücrelerde %100. Bu pilotta Laya ilk sıralara ek başarı getirmedi ve ek gecikme oluşturdu. Laya karşılaştırmada tutuldu.

25 pilot sorgusu geliştirme kümesi olarak ayrıldı. Kalan 304 kaynak grubu için yeni sorgular model sonuçlarına bakılmadan hazırlanıyor; bunlar ayrı değerlendirme kümesi olacaktır. Olası birleşik 329 sorgu özeti keşifsel olarak etiketlenecek; daha önce görülmemiş veri kümesi veya dokunulmamış test sayılmayacaktır. Tarihsel 329 literal sorgu / 434 koşul sonuçları değiştirilmedi.

## Kanallar ve adil karşılaştırma

| Kod | Aday içeriği ve arama yöntemi |
|---|---|
| B | Whisper transkripsiyonu üzerinde BM25 |
| G | Whisper transkripsiyonu üzerinde BGE-M3 |
| E | Whisper transkripsiyonu üzerinde EG2 metin embedding'i |
| N | Sesin kendisinden hazırlanmış EG2 ham ses embedding'i |
| J | Ses + Whisper transkripsiyonundan hazırlanmış EG2 ortak embedding'i |

Bütün sorgular **yazılı Türkçe metindir**. N ve J adayların temsilini belirtir. Laya, ham sesi dinlemez; getirilen adayların Whisper metnini sıralar. Hibritler tablodaki kanalların RRF k=60 birleşimidir. BM25 yalnız pozitif eşleşmeleri kullanır; listeyi yapay belgelerle doldurmaz.

Ana `total` bütçede kanalların retrieval giriş bütçesi toplam 100'dür. Ek `per_channel` bütçede her kanala 100 aday hakkı tanınır. Her iki durumda fusion sonrası en fazla 50 aday alınır. **None ve Laya aynı aday listesiyle eşleşir**; Laya ilk 50'ye girmeyen bir kaydı kurtaramaz. Laya eşiği kapalıdır.

Hit@k, en az bir doğru kayıt ilk k'da mı sorusunu ölçer. Recall@k, o sorguya ait tüm doğru kayıtların ne kadarının ilk k'ya girdiğini ölçer; aynı kaynak grubunda birden çok ses kaydı bulunduğu için bu iki metrik farklıdır.

## Ana karşılaştırma: toplam retrieval bütçesi 100

Süreler gerçek ölçümden gelir; her hücrede 25 örnek vardır. None ve Laya sütunları aynı t0 başlangıcını paylaşır. Tabloda yüzde ve milisaniyeler iki ondalığa yuvarlanmıştır; tam hassasiyet JSON'dadır.

| Kanal | None Hit@5 / @20 (%) | Laya Hit@5 / @20 (%) | None Recall@5 / @20 (%) | Laya Recall@5 / @20 (%) | None p50 / p95 (ms) | Laya p50 / p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| B | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 96.67 / 100.00 | 0.31 / 0.86 | 310.07 / 388.19 |
| G | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 17.15 / 18.71 | 331.28 / 404.59 |
| E | 100.00 / 100.00 | 92.00 / 100.00 | 100.00 / 100.00 | 88.00 / 100.00 | 27.84 / 30.01 | 407.41 / 470.89 |
| N | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 26.94 / 29.43 | 416.77 / 493.02 |
| J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 90.67 / 100.00 | 25.74 / 28.36 | 345.80 / 418.96 |
| B+G | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 90.67 / 100.00 | 17.67 / 19.53 | 412.85 / 501.00 |
| B+E | 100.00 / 100.00 | 92.00 / 100.00 | 100.00 / 100.00 | 88.00 / 100.00 | 28.24 / 31.12 | 400.83 / 483.16 |
| B+N | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 90.67 / 100.00 | 25.94 / 27.75 | 343.45 / 416.83 |
| B+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 89.33 / 100.00 | 27.73 / 29.72 | 407.34 / 502.84 |
| G+E | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 44.91 / 50.49 | 425.64 / 502.07 |
| G+N | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 43.34 / 48.16 | 423.18 / 502.23 |
| G+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 45.57 / 51.33 | 429.56 / 508.49 |
| E+N | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 50.52 / 54.24 | 369.30 / 434.89 |
| E+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 90.67 / 100.00 | 54.94 / 58.09 | 455.00 / 526.63 |
| N+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 55.68 / 62.61 | 453.13 / 521.21 |
| B+G+E | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 91.33 / 100.00 | 45.14 / 47.65 | 393.96 / 445.75 |
| B+G+N | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 44.73 / 49.09 | 432.89 / 514.07 |
| B+G+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 43.42 / 45.91 | 359.92 / 407.53 |
| B+E+N | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 89.33 / 100.00 | 50.27 / 62.09 | 365.50 / 430.62 |
| B+E+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 89.33 / 100.00 | 54.35 / 57.98 | 413.89 / 501.52 |
| B+N+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 54.16 / 60.41 | 437.10 / 509.25 |
| G+E+N | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 71.66 / 82.42 | 441.24 / 516.63 |
| G+E+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 67.84 / 71.91 | 381.65 / 458.65 |
| G+N+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 71.87 / 78.73 | 416.33 / 495.18 |
| E+N+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 85.33 / 93.85 | 476.90 / 590.55 |
| B+G+E+N | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 90.67 / 100.00 | 75.18 / 82.81 | 491.78 / 597.02 |
| B+G+E+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 90.67 / 100.00 | 72.80 / 81.21 | 444.84 / 523.78 |
| B+G+N+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 72.20 / 81.47 | 461.82 / 548.13 |
| B+E+N+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 90.67 / 100.00 | 80.55 / 86.75 | 415.05 / 506.14 |
| G+E+N+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 98.54 / 109.44 | 471.41 / 556.28 |
| B+G+E+N+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 99.79 / 106.72 | 454.99 / 531.83 |

## Ek karşılaştırma: kanal başına retrieval bütçesi 100

Bu bölüm aday bütçesinin etkisini ayrı gösterir; ana toplam bütçeli karşılaştırmayla birleştirilmez. Tek kanallarda aday bütçesi eşdeğer olsa da ayrı gerçek çalıştırmaların gecikmeleri değişebilir.

| Kanal | None Hit@5 / @20 (%) | Laya Hit@5 / @20 (%) | None Recall@5 / @20 (%) | Laya Recall@5 / @20 (%) | None p50 / p95 (ms) | Laya p50 / p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| B | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 96.67 / 100.00 | 0.33 / 0.87 | 325.38 / 396.61 |
| G | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 17.33 / 20.12 | 391.55 / 472.42 |
| E | 100.00 / 100.00 | 92.00 / 100.00 | 100.00 / 100.00 | 88.00 / 100.00 | 27.95 / 33.64 | 411.15 / 526.88 |
| N | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 27.71 / 30.52 | 442.49 / 524.56 |
| J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 90.67 / 100.00 | 28.10 / 30.99 | 436.83 / 523.08 |
| B+G | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 19.02 / 21.46 | 459.57 / 533.80 |
| B+E | 100.00 / 100.00 | 92.00 / 100.00 | 100.00 / 100.00 | 88.00 / 100.00 | 28.64 / 31.59 | 421.58 / 475.41 |
| B+N | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 27.77 / 33.42 | 411.88 / 489.74 |
| B+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 90.67 / 100.00 | 25.07 / 27.27 | 336.87 / 412.43 |
| G+E | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 44.55 / 47.95 | 411.10 / 490.28 |
| G+N | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 45.38 / 51.83 | 424.89 / 498.29 |
| G+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 48.20 / 58.09 | 478.92 / 561.62 |
| E+N | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 58.56 / 63.40 | 490.14 / 561.98 |
| E+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 89.33 / 100.00 | 54.56 / 58.26 | 419.09 / 493.62 |
| N+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 53.98 / 58.53 | 433.74 / 514.11 |
| B+G+E | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.67 / 100.00 | 46.11 / 50.03 | 433.22 / 501.38 |
| B+G+N | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 44.52 / 50.33 | 443.71 / 526.63 |
| B+G+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 90.67 / 100.00 | 43.11 / 46.02 | 358.04 / 432.06 |
| B+E+N | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 89.33 / 100.00 | 55.21 / 61.02 | 439.94 / 514.26 |
| B+E+J | 100.00 / 100.00 | 92.00 / 100.00 | 100.00 / 100.00 | 88.00 / 100.00 | 50.99 / 58.92 | 370.57 / 426.13 |
| B+N+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 60.51 / 80.31 | 544.84 / 689.79 |
| G+E+N | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 73.63 / 80.46 | 459.10 / 565.28 |
| G+E+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 90.67 / 100.00 | 78.82 / 100.19 | 516.87 / 694.57 |
| G+N+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 72.83 / 80.74 | 476.86 / 563.95 |
| E+N+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 82.24 / 89.31 | 467.37 / 529.39 |
| B+G+E+N | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 77.89 / 114.42 | 504.14 / 652.80 |
| B+G+E+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 91.33 / 100.00 | 74.28 / 78.89 | 474.54 / 544.26 |
| B+G+N+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 72.92 / 80.19 | 452.30 / 540.53 |
| B+E+N+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 90.67 / 100.00 | 81.84 / 89.40 | 452.95 / 543.02 |
| G+E+N+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 100.88 / 111.16 | 484.86 / 573.17 |
| B+G+E+N+J | 100.00 / 100.00 | 96.00 / 100.00 | 100.00 / 100.00 | 92.00 / 100.00 | 98.59 / 106.65 | 477.07 / 522.35 |

## Süre, kaynak kullanımı ve yeniden kullanım

- **None = t1−t0:** yeni sorgu embedding'i, arama ve fusion tamamlanana kadar geçen süre.
- **Laya = t2−t0:** aynı t0'dan aynı aday listesinin tamamının yeni Laya skorlarıyla sıralanmasına kadar geçen süre.
- Prefix eşleşmiş iki sonuçta ortaktır; iki sütunu bağımsız iş gibi toplamayın. Aşama p95'lerini toplayarak uçtan uca p95 üretilmez. Metrik hesabı, kayıt ve hash işlemleri t0…t2 dışında tutulur.
- **3.200 yeni sorgu encode işlemi ve 77.266 yeni Laya sorgu–aday skoru** üretildi; sorgu ve skor cache kullanımı sıfırdır. Her encoder için bir yapay ısınma sorgusu ve Laya için bir yapay 50 çift çağrısı kalite örneklerinin dışındadır.
- 743 kaydın daha önce hazırlanmış belge vektörleri ve transkripsiyonları doğrulanarak yeniden kullanıldı. **Yeni belge embedding'i veya ASR çıkarımı yapılmadı.** Bu nedenle N'nin ASR gerektirmeyen çevrimdışı hazırlık avantajını bu yeni sürelerden ölçmüş saymıyoruz. Dosya/cache okuma, çıkarım süresi değildir. Eski cache'in üretim anındaki dosya checksum mühürlemesi bulunmadığı sınırlaması korunur; güncel doğrulama bunu geriye dönük doldurmaz.
- Apple M4 Max / 128 GiB birleşik bellek / macOS 27.0 üzerinde tek süreç, eşzamanlılık 1, MPS kullanıldı. G/E/N/J sorgu batch'i 1, bfloat16; Laya SDK batch'i 8, float32. Modeller aynı süreçte yüklü, çağrılar sıralıdır; E/N/J arasında encode paylaşımı yoktur. Güç/termal geçmişi ölçülmedi.
- Sorgu p50/p95 değerleri modeller yüklendikten ve ısındıktan sonraki çevrimi ölçer. 25 örnekte NumPy doğrusal yüzdelik yöntemi kullanıldı; p95 bir üretim SLA'sı değildir. Sabit hash sırası kullanıldı, başarıya göre sıra seçilmedi.
- Süreç RSS yüksek su seviyesi 5.667.864.576 bayt; örneklenmiş MPS driver maksimumu 7.800.193.024 bayt, MPS allocated maksimumu 5.959.798.272 bayttır. Bunlar toplanmaz; MPS değeri sürekli ölçülmüş cihaz tepe kullanımı değildir.

Ana tek kanal koşullarında BGE-M3'ün sıcak sorgu p50'si yaklaşık 17,15 ms, EG2 yollarınınki 25,74–27,84 ms'dir. Bu pilot EG2'nin sıcak yazılı sorgu yolunun daha hızlı olduğunu göstermiyor. Ham ses belge hazırlığı ile transkripsiyon hazırlığının uçtan uca maliyeti ayrı deney gerektirir. Laya eklendiğinde aynı tek kanal uç noktalarının p50'si yaklaşık 310–417 ms'dir.

Soğuk hazırlık ve yapay ısınma ayrı kaydedildi:

| Kanal | Soğuk model yükleme / BM25 indeksleme (s) | Yapay ısınma (s) |
|---|---:|---:|
| B | 0.162998 | 0.000311 |
| G | 6.079164 | 0.162917 |
| E | 1.260273 | 0.082222 |
| N | 1.652799 | 0.032343 |
| J | 1.277622 | 0.028573 |
| laya | 2.105168 | 0.291721 |

Model constructor süreleri, ortam sürümleri, aşama ölçümleri ve ön/son veri doğrulama süreleri JSON'da ayrı alanlardır. Bu tablo yalnız model yükleme/indeksleme ve ısınmayı gösterir; bütün soğuk başlangıç giderlerinin toplamı değildir.

## Denetlenebilir kayıt

[Tam 124 hücre ve güvenli SHA provenance](../reports/multimodal/fleurs-tr-paraphrase-pilot-v1/summary.json) · [Önceden sabitlenen süre protokolü](fleurs-paraphrase-timing-protocol.tr.md) · [Tarihsel literal test](../reports/multimodal/speech-reranker-summary.md).

Kaynak commit'i `85839b2cfef76f0a0cd29e8f2377ea2b2e0b7065`; [bu commit'in CI'ı başarılı](https://github.com/erendikmenn/rag-benchmark/actions/runs/39853900011). Bağımsız gerçek sonuç incelemesi SHA-256 `bb6ddd8b67d0d544be7bf5c8153784c35b10f964f4c5a8adc5cf5d61fb447648`. Bu inceleme tüm kapalı sonuçları model tekrar çağırmadan hesapladı. Yayında ham sorgu, kaynak metin, ses, özel dosya yolları ve kimlik listeleri bulunmaz. Mevcut modeller yerel çalıştı; ücretli API, yeni GPU kiralama veya yeni model satın alma kullanılmadı.
