# Jev ve GPT 6.1 Sol Ultra: aynı cevaplara verilen notlar

**Aynı 20 sorunun, on yöntemden gelen 200 cevabı değerlendirildi. Cevaplar yeniden üretilmedi.** İstenen `gpt-6.1-sol` modeli ve `ultra` düşünme ayarı, geçmiş konuşmayı almayan üç paralel Codex ajanına açıkça verildi. Bir ajan doğruluğu, iki ayrı ajan kaynak desteğini değerlendirdi. Ajanların notları tamamlanana kadar Jev sonuçları ve yöntem adları gösterilmedi.

Jev değerlendirmesindeki aynı puanlama kuralları ve vaka başına aynı soru, cevap ve kanıt kullanıldı. Birebir aynı girdiler yeniden puanlanmadı: **130 doğruluk + 189 kaynak desteği = 319 benzersiz görev**; bu kararlar 200 cevap için iki ayrı değerlendirmeye dönüştürüldü. Ayrıca altı sentetik örnek için 12 ayrı kontrol kararı alındı. Tüm 331 ajan kararı tamamlandı ve girdi hash’leri, etiketler ve kaynak üyelikleri doğrulandı. Kontroller benchmark paydasına eklenmedi.

## Yöntemlere verilen tam doğru notları

Her hücrenin paydası 20 cevaptır. Tüm cevapları aynı yerel Gemma 4 modeli üretmiştir. Kararsız/değerlendirilemeyen cevaplar paydadan çıkarılmamıştır.

| Arama yöntemi | Laya | Jev tam doğru | Sol Ultra tam doğru | Sol Ultra hem doğru hem kaynakla destekli |
|---|---|---:|---:|---:|
| BM25 | Kapalı | 11/20 (%55) | 14/20 (%70) | 13/20 (%65) |
| BM25 | Açık | 9/20 (%45) | 9/20 (%45) | 9/20 (%45) |
| BGE-M3 | Kapalı | 12/20 (%60) | 12/20 (%60) | 11/20 (%55) |
| BGE-M3 | Açık | 9/20 (%45) | 9/20 (%45) | 9/20 (%45) |
| EmbeddingGemma 2 | Kapalı | 14/20 (%70) | 11/20 (%55) | 10/20 (%50) |
| EmbeddingGemma 2 | Açık | 8/20 (%40) | 9/20 (%45) | 9/20 (%45) |
| BM25 + BGE-M3 | Kapalı | 13/20 (%65) | 13/20 (%65) | 12/20 (%60) |
| BM25 + BGE-M3 | Açık | 8/20 (%40) | 8/20 (%40) | 8/20 (%40) |
| BM25 + EmbeddingGemma 2 | Kapalı | 14/20 (%70) | 14/20 (%70) | 13/20 (%65) |
| BM25 + EmbeddingGemma 2 | Açık | 7/20 (%35) | 8/20 (%40) | 8/20 (%40) |

**BM25 + EmbeddingGemma 2, Laya kapalı:** İki hakem de 14/20 (%70) tam doğru diyor. Ancak yalnızca **11 cevabı birlikte tam doğru** sayıyorlar. Üç cevap yalnızca Jev’e, üç başka cevap yalnızca Sol’a göre tam doğru. Jev’in dağılımı 14 doğru + 6 kısmi; Sol’un dağılımı 14 doğru + 4 kısmi + 2 değerlendirilemez. Sol’a göre hem doğru hem kaynakla tam destekli olanlar 13/20 (%65). Oranların eşit olması kararların aynı olduğu anlamına gelmiyor.

## Hakemler ne kadar uyuşuyor?

| Ölçüm | Aynı etiketi verdikleri / toplam | Uyum |
|---|---:|---:|
| Doğruluk, 200 cevap satırı | 150/200 | %75 |
| Kaynak desteği, 200 cevap satırı | 166/200 | %83 |
| Doğruluk, tekrarlar çıkarılmış görevler | 83/130 | %63,85 |
| Kaynak desteği, tekrarlar çıkarılmış görevler | 158/189 | %83,60 |

Aynı cevap farklı yöntemlerden gelince cevap tablosunda tekrar yer alır. Bu yüzden 200 satırlık uyum ile benzersiz görev uyumu farklıdır. Hiçbiri hakemin insan kararına göre doğruluk oranı değildir. 200 cevap hâlâ yalnızca 20 farklı sorudan gelmektedir.

| Doğruluk etiketi | Jev | Sol Ultra |
|---|---:|---:|
| Tam doğru | 105 | 107 |
| Kısmen doğru | 62 | 50 |
| Yanlış | 3 | 0 |
| Cevap vermekten kaçınmış | 30 | 24 |
| Mevcut kanıtla değerlendirilemez | 0 | 19 |

105/200 ve 107/200, on farklı sistemi bir araya getiren toplam sayılardır; tek bir sistemin kullanım başarısı değildir. Sol’un sıfır `incorrect` etiketi vermesi de hiç hata olmadığı anlamına gelmez: önemli hatalar `partial` içinde, karar verilemeyen örnekler `unjudgeable` içindedir.

## Somut ayrışmalar

- **Beşiktaş sıralaması:** Kelime F1’i 0,25 olan uzun yanıtı iki hakem de doğru ve kaynakla destekli buldu. Farklı ifade, tek başına anlam hatası değil.
- **Amy Winehouse şarkıları:** İstenen iki şarkıdan birini veren yanıtı iki hakem de kısmen doğru ve kaynakla destekli buldu. Eksik cevap ile uydurma bilgi ayrı değerlendiriliyor.
- **Kervansarayın tarihsel gelişimi:** BM25 + EmbeddingGemma cevabını Jev kısmi, Sol tam doğru saydı. Sol, sorunun istediği temel dönem/işlev ve terk edilme bilgilerinin bulunduğunu belirtti.
- **Kariyerin son yılı sorusu:** Sol, kaynakta listelenen son çalışmanın gerçek kariyer bitişini kanıtlamadığını ve düzleşmiş eser listesinin belirsizlik oluşturduğunu işaretledi. Jev aynı örneğe tam doğru dedi. Bu fark yalnızca üretici modelden değil, referans/soru kalitesinden de kaynaklanabilir.
- **Telefon işlemcisinin yanlış modele bağlanması:** Jev kaynak desteğine tam destekli dedi; Sol bunu kısmi destek ve inceleme bayrağıyla değerlendirdi. Sol’un gerekçesi sorunu fark ediyor, ancak önceden yapılan kaynak kontrolündeki açık Z5 teknik bilgisine rağmen `contradicted` etiketi vermiyor. Bu örnekte Sol’un sonucu da kesin doğru hakem kararı olarak kabul edilmedi.
- **İlk sahne oyunu:** Dört cevap satırında Sol, cevapta söylenen başlangıç eseri ile biyografide açıkça belirtilen ilk oyunu çelişkili buldu. Jev bu dört satırın kaynak desteğine tam destekli demişti. Bu, insan kontrolünde öncelik verilebilecek somut bir uyuşmazlık.

Kaynak desteğinde Jev 176 cevabı tam destekli saydı; Sol 148 tam destekli, 24 kısmi destekli, 24 yanıtsız ve 4 çelişkili etiketi verdi. Bu fark daha sıkı değerlendirmeyi gösterir; hangi etiketlerin doğru olduğunu tek başına kanıtlamaz.

## Kontroller ve sınırlar

Altı basit, asistan tarafından yazılmış Türkçe kontrol örneğinin 12 kararında Sol **12/12**, Jev **10/12** beklenen etiketi verdi. Sol, kaynakta bilgi bulunmamasını yanlışlığın kanıtından ayırdı. Bunlar küçük davranış kontrolleridir; insan etiketli bir hakem doğruluk testi değildir. Önceki Jev kontrolündeki geçersiz ilk yanıt ve açık tekrar denemesi eski raporda korunuyor.

Jev, tek tek API istekleriyle; Sol, grupları okuyabilen ve önceki örneklerden bağlam taşıyan Codex ajanlarıyla çalıştı. Grup dosyalarında ortak pasajlar vardı; ajanlara yalnızca her vakada seçilmiş pasajları kullanmaları söylendi. Vaka başına izin verilen girdilerin eşitliği hash ile doğrulandı, ancak bu düzen tek tek izole Sol API çağrılarıyla aynı deney koşulu değildir. Bir doğruluk ajanı ile iki kaynak ajanı kullanıldı; birden çok bağımsız tekrar çalıştırılmadı.

Model ve düşünme ayarı araç çağrısında açıkça seçildi. Bu iş akışı sunucunun kesin model snapshot kimliğini, token kullanımını veya parasal maliyetini döndürmüyor; bunlar uydurulmadı. Yeni Jev/OpenRouter çağrısı yapılmadı. Eski Jev sonuçları değiştirilmedi.

**Bu karşılaştırma, Sol’u otomatik olarak gerçek cevap anahtarı yapmaz.** İnsan kontrolü henüz yapılmadı. Özellikle farklı not verilen örnekler ile rastgele seçilmiş anlaşma örnekleri, hakem adları gizlenerek Türkçe insan değerlendirmesinden geçirilmelidir. Büyük son test hâlâ çalıştırılmadı.

[Sayısal özet](summary.json) · [Her cevap için iki hakemin etiketleri](per-answer-comparison.jsonl) · [İngilizce sayım raporu](report.md) · [Önceki Jev kontrolü](../semantic-controls-tr/report.md)
