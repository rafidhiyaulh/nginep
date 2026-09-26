# Nginep

Cari hotel di Bali pakai bahasa sehari-hari. Ketik yang kamu butuhkan, dapatkan hotel yang diurutkan dengan bukti nyata dari ulasan asli.

Coba sekarang: **https://nginep-610631276830.asia-southeast2.run.app**

## Kenapa project ini dibuat

Saya melamar posisi Data Science di Traveloka. Lowongan mereka minta pengalaman LLM based structured extraction, ranking, dan evaluasi yang benar-benar diukur, bukan cuma slide presentasi. Jadi saya bikin produk yang benar-benar jalan, bukan cuma cerita di slide.

Pencarian hotel biasanya cuma checkbox. Yang orang benar-benar mau adalah sesuatu yang lebih spesifik, seperti 'hotel tenang di Ubud buat kerja remote, wifi kencang'. Baca ratusan ulasan sendiri buat mengecek itu butuh waktu lama. Nginep yang membaca semuanya untukmu, dan menunjukkan kutipan asli sebagai bukti, bukan ringkasan karangan.

## Buat siapa project ini

- Siapa saja yang mau liburan ke Bali dan mau bukti nyata, bukan cuma rating bintang.
- Tim rekrutmen Traveloka: ini jawaban langsung dari lowongan mereka yang benar-benar jalan, bukan cuma ditulis di CV.

Cakupannya masih terbatas ke sejumlah hotel asli, belum ada harga live atau booking.

## Lihat langsung cara kerjanya

**1. ketik yang kamu butuhkan**

<img src="docs/images/01-search-empty.png" alt="Halaman pencarian" width="520" />

**2. dapat bukti, bukan tebakan**

<img src="docs/images/02-search-results.png" alt="Hasil terurut dengan kutipan asli" width="520" />

## Angka-angkanya

| Yang diukur | Hasil |
|---|---|
| Akurasi ekstraksi aspek vs benchmark manusia (HoASA) | 91,9% kesepakatan, 0,856 F1 |
| Kualitas ranking (NDCG@10), skor berbobot | 0,992 |
| Sama, diurutkan cuma berdasar rating bintang | 0,867 |
| Sama, pencarian kata kunci | 0,753 |
| Sama, cuma kemiripan embedding | 0,798 |
| Ranker LightGBM, cross validated | 0,984, tidak menang |

Baris terakhir dilaporkan apa adanya, tidak disembunyikan. Detail lengkap ada di `reports/eval/`.

## Keterbatasan yang diketahui

- Cuma 16 hotel yang punya data ulasan asli. Hotel lain tetap muncul untuk areanya, tapi tidak diberi ranking.
- Ulasannya berbahasa Inggris, query bisa bahasa Indonesia. Ini memang disengaja.
- Pencocokan area berbasis teks, bukan geografi asli.
- Belum ada harga live atau booking.
- Label untuk evaluasi ranking dibantu Claude, bukan diverifikasi manusia secara independen. Lihat `labeling/LABELING_METHOD.md`.

## Selengkapnya

Detail teknis (arsitektur, setup, API) ada di `docs/technical.md`. Lisensi kode: MIT. Lisensi data ada di `data/raw/SOURCES.md`.
