# Türkçe konuşma: kalan 304 yazılı paraphrase sorgusunun sonuçları

Ayrı **304 sorguluk mevcut değerlendirme kohortunun tamamı** aynı **743 ses kaydı** üzerinde tamamlandı: beş kanalın 31 boş olmayan alt kümesi × iki retrieval bütçesi × None/Laya, toplam **124 hücre ve 37.696 sorgu-hücre sonucu**. Önceki 25 sorguluk pilot korunur; bu rapor yalnız kalan 304 sorguyu kapsar ve birleşik 329 sorgu metriği üretmez.

Bağımsız gerçek çıktı incelemesi geçti; süreç gerçek bekleme kaydıyla **exit 0** verdi. Tam koşu, aşağıda açıklanan ortak kalite iş akışıdır. **Hücre başına yeni gecikme ölçümü yapılmadı.**

## Kapsam ve etiketler

304 yeni Türkçe yazılı sorgu, özgün 329 kaynak grubundan 25 pilot grubunun çıkarılmasıyla kalan grupları kapsar. Sorguların tamamı ajan anlam incelemesinden geçti; kaynak gruplarından **689 pozitif sorgu–kayıt etiketi** aktarıldı. **İnsan doğrulaması ve galerinin tamamında semantik uygunluk etiketlemesi yapılmadı.** Etiketler keşifsel kaynak grubu eşleşmeleridir; ilgili olabilecek başka kayıtların tamamını bildiğimizi varsaymayız.

Bütün 329 kaynak grubu daha önce literal sorgularla görülmüştü. Kalan 304'ün tamamlanması bu veri setini dokunulmamış test yapmaz. Bulgular bu sorgu kohortu, galeri, etiket aktarımı ve arama protokolüyle sınırlıdır; genel model kazananı ilan edilmez. Cevap üretimi, konuşulmuş sorgu, gürültü dayanıklılığı ve uzun konuşma performansı ölçülmedi. Güven aralığı veya p-değeri hesaplanmadı.

## Kanallar ve karşılaştırma

| Kod | Aday içeriği ve arama yöntemi |
|---|---|
| B | Whisper transkripsiyonu üzerinde BM25 |
| G | Whisper transkripsiyonu üzerinde BGE-M3 |
| E | Whisper transkripsiyonu üzerinde EG2 metin embedding'i |
| N | Sesin kendisinden hazırlanmış EG2 native ses embedding'i |
| J | Ses + Whisper transkripsiyonundan hazırlanmış EG2 ortak embedding'i |

**Bütün sorgular yazılı Türkçe metindir.** N ve J galerideki aday temsilini belirtir. Laya yalnız adayların Whisper metnini puanlar; sesi dinlemez. Hibritler eşit ağırlıklı RRF k=60 birleşimidir. BM25 yalnız pozitif eşleşmeleri kullanır ve listeyi yapay kayıtlarla doldurmaz.

Ana `total` koşulunda toplam retrieval giriş bütçesi 100, ek `per_channel` koşulunda her kanalın giriş bütçesi 100'dür. Birleşimden sonra en fazla 50 aday tutulur. None ve Laya her sorgu/alt küme/bütçe için aynı aday listesini kullanır; Laya bu liste dışındaki kaydı getiremez. Laya eşiği kapalıdır. Gerçekte tek başına BM25 sorgu başına 3–50 aday, dört dense tek kanal ise her sorguda 50 aday getirdi. Bazı ana toplam-bütçeli hibritlerde de 50'den az aday vardı; hücrelerin min/max/toplam aday sayıları JSON'da korunur.

Hit@k ilk k'da en az bir etiketli doğru kayıt bulunmasını, Recall@k ise sorgunun tüm etiketli doğru kayıtlarının ne kadarının ilk k'da bulunduğunu ölçer. Metrikler sorgular üzerinden makro ortalamadır. Tablodaki yüzdeler iki ondalığa yuvarlandı; [JSON'daki 124 hücre](../reports/multimodal/fleurs-tr-paraphrase-remaining304-v1/summary.json) bütün 17 metriği tam kayan nokta hassasiyetiyle korur.

## Gözlenen bulgular

Ana toplam bütçede tek kanallar:

- **B:** None → Laya ile Hit@5 %98.36 → %98.68; Recall@5 %97.97 → %97.92.
- **G:** None → Laya ile Hit@5 %100.00 → %99.01; Recall@5 %100.00 → %98.30.
- **E:** None → Laya ile Hit@5 %100.00 → %98.68; Recall@5 %100.00 → %97.92.
- **N:** None → Laya ile Hit@5 %99.67 → %98.68; Recall@5 %98.85 → %98.14.
- **J:** None → Laya ile Hit@5 %100.00 → %98.68; Recall@5 %100.00 → %98.14.

BM25'te Laya Hit@5'i artırırken Recall@5 biraz düştü; G/E/N/J tek kanallarının tamamında her iki @5 metriği de düştü. Laya olmadan G/E/J tek kanalları tablolarda gösterilen dört ölçümde de %100'e ulaştı. Bu kohortta da metrikler büyük ölçüde doygundur; küçük farklar genel bir üstünlük kanıtı değildir.

31 alt kümenin gözlenen aralıkları:

- **ana total, None:** Hit@5 %98.36–100.00; Hit@20 %99.34–100.00; Recall@5 %97.97–100.00; Recall@20 %99.18–100.00.
- **ana total, Laya:** Hit@5 %98.68–99.01; Hit@20 %99.34–100.00; Recall@5 %97.92–98.57; Recall@20 %99.01–99.84.
- **ek per_channel, None:** Hit@5 %98.36–100.00; Hit@20 %99.34–100.00; Recall@5 %97.97–100.00; Recall@20 %99.18–100.00.
- **ek per_channel, Laya:** Hit@5 %98.68–99.34; Hit@20 %99.34–100.00; Recall@5 %97.92–98.74; Recall@20 %99.01–99.84.

ana total bütçesinde Laya ile Hit@5 1 alt kümede arttı, 30 alt kümede düştü, 0 alt kümede aynı kaldı; Recall@5 0 alt kümede arttı, 31 alt kümede düştü, 0 alt kümede aynı kaldı.
ek per_channel bütçesinde Laya ile Hit@5 1 alt kümede arttı, 30 alt kümede düştü, 0 alt kümede aynı kaldı; Recall@5 0 alt kümede arttı, 31 alt kümede düştü, 0 alt kümede aynı kaldı. Bu sayımlar betimseldir; birbirine örtüşen alt kümeler bağımsız deney tekrarları değildir.

## Ana karşılaştırma: toplam retrieval bütçesi 100

| Kanal | None Hit@5 / @20 (%) | Laya Hit@5 / @20 (%) | None Recall@5 / @20 (%) | Laya Recall@5 / @20 (%) |
|---|---:|---:|---:|---:|
| B | 98.36 / 99.34 | 98.68 / 99.34 | 97.97 / 99.18 | 97.92 / 99.01 |
| G | 100.00 / 100.00 | 99.01 / 100.00 | 100.00 / 100.00 | 98.30 / 99.73 |
| E | 100.00 / 100.00 | 98.68 / 100.00 | 100.00 / 100.00 | 97.92 / 99.84 |
| N | 99.67 / 100.00 | 98.68 / 100.00 | 98.85 / 100.00 | 98.14 / 99.84 |
| J | 100.00 / 100.00 | 98.68 / 100.00 | 100.00 / 100.00 | 98.14 / 99.84 |
| B+G | 99.67 / 100.00 | 99.01 / 100.00 | 99.51 / 100.00 | 98.46 / 99.84 |
| B+E | 99.67 / 100.00 | 98.68 / 100.00 | 99.51 / 100.00 | 98.14 / 99.84 |
| B+N | 99.67 / 100.00 | 98.68 / 100.00 | 99.51 / 100.00 | 98.25 / 99.84 |
| B+J | 99.67 / 100.00 | 98.68 / 100.00 | 99.51 / 100.00 | 98.14 / 99.84 |
| G+E | 100.00 / 100.00 | 98.68 / 100.00 | 100.00 / 100.00 | 98.03 / 99.84 |
| G+N | 100.00 / 100.00 | 98.68 / 100.00 | 99.78 / 100.00 | 98.30 / 99.84 |
| G+J | 100.00 / 100.00 | 98.68 / 100.00 | 100.00 / 100.00 | 98.14 / 99.84 |
| E+N | 100.00 / 100.00 | 98.68 / 100.00 | 99.78 / 100.00 | 98.03 / 99.84 |
| E+J | 100.00 / 100.00 | 98.68 / 100.00 | 100.00 / 100.00 | 98.03 / 99.84 |
| N+J | 100.00 / 100.00 | 98.68 / 100.00 | 99.67 / 100.00 | 98.30 / 99.73 |
| B+G+E | 100.00 / 100.00 | 98.68 / 100.00 | 100.00 / 100.00 | 98.14 / 99.73 |
| B+G+N | 100.00 / 100.00 | 99.01 / 100.00 | 100.00 / 100.00 | 98.57 / 99.84 |
| B+G+J | 100.00 / 100.00 | 99.01 / 100.00 | 100.00 / 100.00 | 98.30 / 99.84 |
| B+E+N | 99.67 / 100.00 | 98.68 / 100.00 | 99.67 / 100.00 | 98.14 / 99.73 |
| B+E+J | 99.67 / 100.00 | 98.68 / 100.00 | 99.67 / 100.00 | 98.14 / 99.84 |
| B+N+J | 99.67 / 100.00 | 98.68 / 100.00 | 99.67 / 100.00 | 98.30 / 99.84 |
| G+E+N | 100.00 / 100.00 | 99.01 / 100.00 | 100.00 / 100.00 | 98.46 / 99.84 |
| G+E+J | 100.00 / 100.00 | 99.01 / 100.00 | 100.00 / 100.00 | 98.46 / 99.84 |
| G+N+J | 100.00 / 100.00 | 99.01 / 100.00 | 100.00 / 100.00 | 98.46 / 99.84 |
| E+N+J | 100.00 / 100.00 | 98.68 / 100.00 | 100.00 / 100.00 | 98.14 / 99.84 |
| B+G+E+N | 100.00 / 100.00 | 98.68 / 100.00 | 100.00 / 100.00 | 98.14 / 99.84 |
| B+G+E+J | 100.00 / 100.00 | 98.68 / 100.00 | 100.00 / 100.00 | 98.14 / 99.84 |
| B+G+N+J | 100.00 / 100.00 | 99.01 / 100.00 | 100.00 / 100.00 | 98.46 / 99.84 |
| B+E+N+J | 100.00 / 100.00 | 98.68 / 100.00 | 99.89 / 100.00 | 98.14 / 99.73 |
| G+E+N+J | 100.00 / 100.00 | 99.01 / 100.00 | 100.00 / 100.00 | 98.46 / 99.84 |
| B+G+E+N+J | 100.00 / 100.00 | 98.68 / 100.00 | 100.00 / 100.00 | 98.14 / 99.84 |

## Ek karşılaştırma: kanal başına retrieval bütçesi 100

Bu tablo bütçe değişimini ayrı gösterir; ana toplam bütçeli sonuçlarla birleştirilmez.

| Kanal | None Hit@5 / @20 (%) | Laya Hit@5 / @20 (%) | None Recall@5 / @20 (%) | Laya Recall@5 / @20 (%) |
|---|---:|---:|---:|---:|
| B | 98.36 / 99.34 | 98.68 / 99.34 | 97.97 / 99.18 | 97.92 / 99.01 |
| G | 100.00 / 100.00 | 99.01 / 100.00 | 100.00 / 100.00 | 98.30 / 99.73 |
| E | 100.00 / 100.00 | 98.68 / 100.00 | 100.00 / 100.00 | 97.92 / 99.84 |
| N | 99.67 / 100.00 | 98.68 / 100.00 | 98.85 / 100.00 | 98.14 / 99.84 |
| J | 100.00 / 100.00 | 98.68 / 100.00 | 100.00 / 100.00 | 98.14 / 99.84 |
| B+G | 99.67 / 99.67 | 99.34 / 100.00 | 99.51 / 99.67 | 98.74 / 99.84 |
| B+E | 99.67 / 99.67 | 98.68 / 100.00 | 99.51 / 99.67 | 98.14 / 99.84 |
| B+N | 99.67 / 99.67 | 98.68 / 100.00 | 99.51 / 99.67 | 98.25 / 99.84 |
| B+J | 99.67 / 99.67 | 98.68 / 100.00 | 99.51 / 99.67 | 98.14 / 99.84 |
| G+E | 100.00 / 100.00 | 98.68 / 100.00 | 100.00 / 100.00 | 98.14 / 99.84 |
| G+N | 100.00 / 100.00 | 98.68 / 100.00 | 99.78 / 100.00 | 98.30 / 99.62 |
| G+J | 100.00 / 100.00 | 98.68 / 100.00 | 100.00 / 100.00 | 98.14 / 99.84 |
| E+N | 100.00 / 100.00 | 98.68 / 99.67 | 99.78 / 100.00 | 98.30 / 99.51 |
| E+J | 100.00 / 100.00 | 98.68 / 100.00 | 100.00 / 100.00 | 98.14 / 99.84 |
| N+J | 100.00 / 100.00 | 98.68 / 100.00 | 99.67 / 100.00 | 98.30 / 99.73 |
| B+G+E | 99.67 / 100.00 | 98.68 / 100.00 | 99.67 / 100.00 | 98.14 / 99.84 |
| B+G+N | 99.67 / 100.00 | 98.68 / 99.67 | 99.67 / 100.00 | 98.14 / 99.51 |
| B+G+J | 99.67 / 100.00 | 98.68 / 100.00 | 99.67 / 100.00 | 98.14 / 99.84 |
| B+E+N | 99.67 / 100.00 | 98.68 / 100.00 | 99.67 / 100.00 | 98.14 / 99.84 |
| B+E+J | 99.67 / 100.00 | 98.68 / 100.00 | 99.67 / 100.00 | 98.14 / 99.84 |
| B+N+J | 99.67 / 100.00 | 98.68 / 100.00 | 99.67 / 100.00 | 98.14 / 99.73 |
| G+E+N | 100.00 / 100.00 | 98.68 / 99.67 | 100.00 / 100.00 | 98.30 / 99.51 |
| G+E+J | 100.00 / 100.00 | 98.68 / 100.00 | 100.00 / 100.00 | 98.14 / 99.84 |
| G+N+J | 100.00 / 100.00 | 98.68 / 99.67 | 100.00 / 100.00 | 98.30 / 99.51 |
| E+N+J | 100.00 / 100.00 | 98.68 / 100.00 | 100.00 / 100.00 | 98.30 / 99.73 |
| B+G+E+N | 100.00 / 100.00 | 98.68 / 100.00 | 99.78 / 100.00 | 98.14 / 99.73 |
| B+G+E+J | 99.67 / 100.00 | 98.68 / 100.00 | 99.67 / 100.00 | 98.14 / 99.84 |
| B+G+N+J | 100.00 / 100.00 | 98.68 / 100.00 | 99.78 / 100.00 | 98.14 / 99.84 |
| B+E+N+J | 99.67 / 100.00 | 98.68 / 100.00 | 99.67 / 100.00 | 98.30 / 99.84 |
| G+E+N+J | 100.00 / 100.00 | 98.68 / 100.00 | 100.00 / 100.00 | 98.14 / 99.62 |
| B+G+E+N+J | 100.00 / 100.00 | 98.68 / 100.00 | 99.89 / 100.00 | 98.14 / 99.84 |

## Gerçek iş akışı ve süre sınırı

Her sorgu için G/E/N/J kanallarında **dört yeni query encode** yapıldı: toplam **1.216**. 62 aday listesindeki sorgu–belge çiftlerinin birleşimi çıkarıldı; bu birleşimdeki her çift Laya ile bir kez yeni puanlandı, sonra aynı skorlar aday kimliğine göre ilgili listelere doğrudan eşlendi. Toplam **47.184 yeni tekil Laya çifti**, listeler arasında **940.505 aday listesi içi sorgu–belge çifti** ve **893.321 aynı sorgu içi skor yeniden kullanımı** kaydedildi. Eşit skorların sırası aday giriş sırasını korur.

Önceki 25 pilotun query veya skor cache'i kullanılmadı. Önceden hazırlanmış 743 kaydın belge vektörleri ve Whisper transkripsiyonları doğrulanarak yeniden kullanıldı; **yeni ASR veya belge embedding çıkarımı sıfırdır**. Eski belge vektörlerinin üretim anında checksum mühürlemesi bulunmuyordu; bugünkü doğrulama bunu geriye dönük sağlamaz. Cache okuma model çıkarımı sayılmaz.

Gerçek tek başlatmanın bütün iş akışı **414.250402667 saniye** sürdü (yaklaşık 6.90 dakika). Bu süre veri doğrulama, model yükleme/ısınma ve kayıt işlemlerini de içeren bu ortak kalite çalışmasının gerçekleşen süresidir. **Çevrimiçi bir arama yolunun veya None/Laya hücrelerinin gecikmesi değildir.** Ortak grafın süresini 124 hücreye bölerek, aşamaları toplayarak ya da bu iş akışını gerçek çevrimiçi boru hattıyla karşılaştırarak gecikme türetilmez.

Adil, eşleşmiş None/Laya sıcak sorgu p50/p95 ölçümleri ayrı [25 sorguluk pilotta](fleurs-paraphrase-pilot-results.tr.md) yer alır; bu 304 sorgu koşusuna taşınmaz veya onunla birleştirilmez. Buradaki birleşim üzerinden pointwise batch düzeni pilotun liste başına batch düzeninden farklıdır; bit düzeyinde skor eşitliği iddiası da yoktur. Yerel MPS üzerinde modeller aynı süreçte yüklü, sorgu aşamaları sıralı ve eşzamanlılık 1'dir. E/N/J arasında query encode paylaşılmaz. Ücretli API, yeni kiralık GPU veya model satın alma kullanılmadı.

## Denetlenebilir yayın

[Tam toplulaştırma ve güvenli SHA kaydı](../reports/multimodal/fleurs-tr-paraphrase-remaining304-v1/summary.json) · [Korunan 25 sorguluk pilot](fleurs-paraphrase-pilot-results.tr.md) · [Tarihsel literal sorgu sonuçları](../reports/multimodal/speech-reranker-summary.md).

Gerçek sonuç incelemesi SHA-256 `bf25a7d196637bacb27556d3881ac6930cb9e73cc06630e99de3c734702b7d23`; bu inceleme kapalı çıktıları model tekrar çağırmadan doğruladı. Yayın yalnız allowlist ile seçilmiş toplulaştırmaları ve tekil SHA provenance alanlarını içerir. Ham soru, transkript, cevap, ses, özel dosya yolu, kimlik listesi veya özel bağımlılık haritası yayımlanmaz. Tarihsel pilot, RAGTurk ve Gemma tabloları korunur.
