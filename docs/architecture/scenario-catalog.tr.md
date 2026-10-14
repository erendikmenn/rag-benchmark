# Senaryo kataloğu ve deney seçimi

14 Ekim 2026. Bu katalog gelecekte seçilebilecek deneyleri tanımlar; hepsini şimdi çalıştırma talimatı veya sınırsız kombinasyon sözü değildir. [Ana mimari](modular-benchmark.tr.md). Mevcut ana matris ve ayrı CUDA sonuçları kendi sürümlerinde korunur.

## Kontrollü karşılaştırmalar

| Soru | Sabit kalanlar | Değişen bileşen | Ölçülecek sonuç |
|---|---|---|---|
| Yeni embedding daha iyi mi? | Corpus/query/split, uygun ortak temsil, arama ve aday bütçesi | Embedding modeli ve gerekli model-specific query/doc arayüzü | Mevcut graded retrieval protokolünde 17 metrik; başka görevde uygun evaluator, model-input kapsamı, cold/warm/cache süreleri, alan |
| Hybrid işe yarıyor mu? | Kaynak kanal sıralamaları ve galeri | B; G; E; B+G; B+E; G+E; B+G+E | RRF bütçesiyle kalite; her yeni birleşim için model çıkarımı gerekmeyebilir |
| BM25 yerine BM25+? | Tokenizasyon, corpus/query ve arama cutoff | Sözcüksel algoritma/formülü | Arama farkı; algoritma isimleri aynı baseline sayılmaz |
| Laya, Jev veya Clef? | Aynı sorgu–aday havuzu, girdi görünümü, alaka sorusu ve score mapping | Decision scorer/provider | Paired ranking farkları, hata/kapsam, gecikme/ücret; kalibrasyon ayrı |
| Cross-encoder mı decision model mi? | Aynı adaylar ve kaynak bütçesi | Reranker ailesi | Ortak ranking çıktısı üzerinden kalite/maliyet; farklı skorları eşit olasılık varsaymama |
| Listwise LLM daha iyi mi? | Aday havuzu, tanımlı aday sırası/pencere politikası | Listwise adapter/model | Aday kaybı/çoğaltma, sıralama kalitesi, sıra duyarlılığı ve maliyet |
| Başka cevap LLM'i? | Aynı sıralanmış kaynak paketi, kaynak ID'leri, cevap görevi | Generator/model/prompt adaptörü | Cevap doğruluğu, citation/entailment, abstention, normal-stop, maliyet |
| Reranker son cevabı iyileştiriyor mu? | Dataset, generator, evidence policy | Reranker none/A/B ve değişen seçilmiş kaynaklar | Retrieval etkisi ile uçtan uca cevap etkisini ayrı raporlama |
| Offline yerine API? | Korunabilen görev/girdiler ve açık model koşulları | Yerel/provider serving | Kalite, tekrar üretilebilirlik, ağ/ücret, rate limit; farklı modeller donanım hız testi değildir |
| Sayfa görüntüsü mü metin mi? | Kaynak sayfa ve sorgular | OCR/native image/generated description/joint | Modaliteye özgü kapsam ve kalite; caption doğruluğu ayrıca |
| Ses/video nasıl temsil edilmeli? | Kaynak dosyası/query/qrels | Native, ASR, caption, frame sampling, audio+video | Kaynak retention, retrieval, ASR/caption kalite ve maliyet; ASR kaynağı gold'u görmez |
| Kodda uzun girdiler? | Kaynak fonksiyon/query ve source ID | Whole-function veya kayıpsız chunk/source-max | Token kapsamı, source ranking; chunk-level sonucu function-level sanmama |
| Daha büyük bağlam/topK? | Model, corpus, query ve belirli retrieval sırası | K, source budget, context policy | Kaynak kapsamı, citation, kalite/maliyet; bağlam aşımı ayrı failure |
| Daha hızlı serving? | Tam aynı örnek ve model/request | Batch, dtype, quantization, motor veya GPU | Önce sayısal/ranking eşdeğerlik; sonra valid örneklerde throughput ve bellek |
| Exact arama yerine ANN? | Aynı vektörler, query ve galeri | İndeks/search parametreleri | Exact topK'ya göre recall, görev metriği, gecikme ve indeks maliyeti |
| Yeni veri/dil/domain? | Model/pipeline protokolü | Dataset veya split | Dataset başına kalite/kapsam; farklı galerilerin skorlarını kör havuzlamama |
| Adaptif pipeline? | Sabit routing politikası ve geliştirme split'i | Query türüne göre retriever/scorer/LLM seçimi | Uçtan uca kalite/maliyet, routing hataları, seçilen yolun provenance'ı |
| Yeni değerlendirme modeli? | Aynı kaydedilmiş cevap ve kaynaklar | Judge/rubric/insan referansı | İnsan uyumu, bias, kalibrasyon; judge skoru insan doğruluğu sayılmaz |

`B/G/E` yukarıdaki hybrid örneğinde BM25/BGE/EG2'yi ifade eder; yeni registry bu harflere yeni modelleri gizlice atamaz. İsimler model/uygulama revision'ına çözümlenir. Aynı görev için bir modelin farklı gerekli instruction'ları varsa model-native ve mümkünse matched-input karşılaştırması ayrı tanımlanır; tek faktör değişti iddiası buna göre sınırlandırılır.

## Deney patlamasını kontrol etmek

On veri kümesi × beş embedding × üç fusion × dört reranker × dört LLM × üç K = 7.200 görünen koşul üretebilir. Bu çarpım otomatik gerçek plan değildir: bazı seçimler geçersizdir, bazıları aynı çıkarımı paylaşır ve bazıları bilimsel soruya katkı sağlamaz. Planlayıcı hem koşulları hem benzersiz vektör/pair/QA isteklerini gösterir; bunları aynı sayı diye sunmaz.

Önerilen dalgalar:

1. **Sabit referans:** Tek veri görünümü, mevcut model kombinasyonu, ölçüm ve eski cache bağlarının doğrulanması.
2. **Bileşen taraması:** Generator çalıştırmadan embedding/fusion/reranker karşılaştırması; aynı QA girdisinde generator karşılaştırması ayrı.
3. **Seçilmiş uçtan uca koşullar:** Geliştirme split'inde seçilen kalite/maliyet Pareto adayları; seçme kuralı ve sınırlar önceden yazılır.
4. **Kilitlenmiş test:** Final split üzerinde aynı plan; test sonucuna bakıp tuning döngüsüne geri dönülürse yeni keşifsel çalışma diye etiketlenir.
5. **Genellenebilirlik ve dayanıklılık:** Dil/domain, uzunluk, eksik modalite, çelişkili kaynak, yanıtsız soru ve kesinti senaryoları. Değiştirilmiş veriler yeni dataset revision'ıdır.

Tam faktöriyel tarama ayrıca seçilebilir. Önceden gereken benzersiz iş, storage ve API bütçesi hesaplanır; desteklenmeyen koşullar görünür kalır. Kaliteyi düşüren sonuçlar ve başarısız denemeler seçilerek rapordan çıkarılmaz. Hızlı geliştirme altkümesi tam test sonucu sayılmaz.

## Çalıştırma ortamı senaryoları

| Ortam | İş dağılımı | Gerekli sınır |
|---|---|---|
| Yalnız Mac | CPU veri/indeks/audit, uyumlu tek MPS model worker | Unified memory paylaşımı ve gerçek input-length profili |
| Mac + tek kiralık GPU | Mac coordinator/CPU işler; GPU ağır inference; kalıcı özel depo | İnternet kopunca lease/commit ve checkpoint; yeniden bağlanma ile otomatik çift iş yok |
| Birden fazla bağımsız GPU | Aynı model replikalarına query/pair shard veya farklı model aileleri | Ortak kaynak/plan kimliği, worker başına sahiplik ve merge doğruluğu |
| Tek çoklu-GPU sunucu | Sığan modeller için GPU başına replica; gerekli büyük model için tensor parallel | Topoloji, VRAM ve motor desteği ölçülür; iki yöntem karıştırılmaz |
| Yerel + API | Yerelde embedding/rerank, API'de generator/decision; veya tersi | Sağlayıcıya izinli kaynaklar, RPM/TPM/ücret ve sürüm kimliği |
| Aralıklı Colab | Tek GPU worker ve sık kapalı shard yedeği | VM kalıcı storage veya kesintisiz cluster düğümü varsayılmaz |
| AWS havuzu | GPU container worker'ları, kalıcı object store, queue | Bölgesel kota, hesap/bütçe seçimi, idle kapatma; bu tasarım tahsis yapmaz |

H100'e atanan “20 iş” kuyrukta 20 shard olabilir. Kaçının aynı anda inference yapacağı uzunluğa, model belleğine, batch/serving motoruna ve ölçülmüş kapasiteye bağlıdır. Kuyruk önceliği büyük/küçük işleri aç bırakmamalı; modeli sürekli yükleyip boşaltma maliyeti de hesaba katılmalıdır.

## Her planın kullanıcıya göstereceği önizleme

- Veri sürümleri, split, query/gallery sayısı ve erişim engelleri.
- Tam model kimlikleri, temsil, girdi sınırları, prompt/decoding, fusion/K ve karşılaştırmada değişen faktörler.
- Planlanan koşullar; desteklenmeyen/dışlanan koşullar ve gerekçeleri.
- Yeniden kullanılabilir artifact'ler; yeni belge/sorgu vektörleri, benzersiz rerank çiftleri ve QA istekleri.
- CPU/GPU/API iş haritası; veri aktarımı ve kalıcı disk gereksinimi.
- Ölçülmüş hızın cihaz, uzunluk dilimi, cache/fresh ve valid/failure kapsamı; bulunmuyorsa “ölçülmedi”.
- Tamamlanmış, doğrulanmış, yayımlanmış kapsam ayrı; son checkpoint sonrası risk altındaki iş miktarı.
- API/GPU maliyet sınırı, durdurma politikası, insan etiketi veya resmî veri erişimine bağlı engeller.

Toplam süre hesabı DAG'ın kritik yoluna, worker başına ölçülen kalan işlere, kurulum/aktarım, sınırlı retries ve audit'e dayanmalı. Örnek karışımı üzerinden GPU-saat hesabı doğrudan başarılı bitiş ETA'sı olmaz. Yeni mimarinin hızlanması henüz ölçülmedi. CIRR resmî medya erişimi ve insan semantic/citation kalibrasyonu daha fazla GPU ile tamamlanmış sayılmaz.
