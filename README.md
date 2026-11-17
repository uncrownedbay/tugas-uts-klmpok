# tugas-uts-kelmpok
## DETEKSI SMS SPAM PROMO DAN PENIPUAN

Klasifikasi Pesan SMS ke dalam Kategori Normal, Promo, dan Penipuan Menggunakan

Machine Learning

Tugas Mata Kuliah Artificial Intelligence

Disusun oleh:

        Henry Ersya Tryas Putra (2501010237) 
        I Wayan Bayu Kusumadinatha (2501010359)
        Andi Pratama (2501010092)

Kotak masuk SMS hari ini tidak hanya berisi pesan dari teman atau keluarga. Di sana

juga ada tawaran paket data, promo operator, dan tidak sedikit pesan penipuan, misalnya

“Selamat, Anda memenangkan mobil! Klik link berikut untuk mengambil hadiah Anda.” Kalau

jumlah pesannya banyak, memilah satu per satu jelas merepotkan, dan pesan penipuan yang

terlewat bisa merugikan penerimanya. Karena itu, laporan ini membahas penerapan machine

learning untuk membaca isi SMS lalu mengelompokkannya ke dalam tiga kategori: Normal,

Promo, dan Penipuan.

Satu catatan penamaan: di dalam kode program, ketiga kategori diberi nama normal,

penipuan, dan promo, sehingga nama kategori di laporan ini sama dengan yang tampil pada

keluaran program.

## 1. Dataset

## 1.1 Pengertian Dataset

Dataset adalah kumpulan data yang disiapkan untuk dipelajari atau diuji oleh sebuah

program. Dalam proyek ini, dataset berperan seperti bahan latihan soal. Model diberi banyak

contoh SMS lengkap dengan “kunci jawabannya”, yaitu kategori pesan tersebut, lalu dari situ

model mencari pola kata yang membedakan satu kategori dengan kategori lain. Karena setiap

contoh sudah memiliki label, pendekatan ini disebut supervised learning.

## 1.2 Jenis Data yang Digunakan

Data yang dipakai berupa teks pesan SMS berbahasa Indonesia. Datanya tersimpan

dalam berkas CSV bernama dataset_sms_spam_v1.csv dengan dua kolom, yaitu teks (isi SMS)

dan label (kategori SMS dalam bentuk angka). Berkas mentahnya berisi 1.143 baris, dan

pengecekan awal menunjukkan tidak ada nilai kosong pada kedua kolom tersebut (hasil

isnull().sum() bernilai 0 untuk teks dan label).

1


Teks SMS termasuk data tidak terstruktur, sehingga tidak bisa langsung dihitung oleh

komputer. Selain itu, gaya tulisnya cukup beragam. Banyak pesan memakai singkatan dan

ejaan tidak baku, seperti “yg”, “utk”, “dptkan”, dan “slamat”. Kondisi ini perlu diingat karena

nanti berpengaruh pada cara model membaca kata.

Dari isinya, SMS Promo dan Penipuan banyak yang mengatasnamakan operator seluler

seperti Telkomsel, Indosat, dan Tri, sedangkan SMS Normal berupa percakapan sehari-hari,

misalnya urusan kuliah dan pekerjaan. Panjang pesannya juga berbeda. Setelah pembersihan,

SMS Normal rata-rata hanya sekitar 12 kata, sedangkan SMS Promo dan Penipuan sekitar 21

kata.

## 1.3 Pembagian Data ke dalam Tiga Kategori

Setiap SMS dimasukkan ke salah satu dari tiga kategori. Di program, kategori disimpan

sebagai angka 0, 1, dan 2, lalu diubah kembali menjadi nama kategori ketika hasilnya

ditampilkan (label_map). Pembagian dan ciri tiap kategori dirangkum pada Tabel 1. Ciri

tersebut disusun dari contoh isi dataset, dan batas antarkategori pada data nyata tidak selalu

tegas, seperti dibahas di bagian 5.

*Tabel 1. Kategori SMS*

| Kategori | Label di program | Ciri pesan |
| --- | --- | --- |
| Normal | 0 (normal) | Pesan percakapan sehari-hari, misalnya urusan kuliah dan pekerjaan. Ada juga beberapa notifikasi dari operator. |
| Penipuan | 1 (penipuan) | Pesan yang mengarah pada penipuan, seperti hadiah palsu dengan PIN pemenang dan tautan mencurigakan, permintaan kirim pulsa atau uang dengan mengaku sebagai keluarga, serta tawaran togel atau agen pulsa. |
| Promo | 2 (promo) | Penawaran produk atau layanan, seperti paket data, kuota, bonus, dan diskon. |

## 1.4 Contoh Data

Tabel 2 menampilkan beberapa contoh SMS asli dari dataset untuk tiap kategori.

Penulisannya dibiarkan sesuai aslinya, termasuk huruf besar dan ejaan tidak baku. Huruf kecil

baru diterapkan saat pembersihan data di bagian 4.3.

*Tabel 2. Contoh SMS per kategori*

| Kategori | Contoh SMS |
| --- | --- |
| Normal | “Jangan gitu bro, cepat ripep lah.” |
| Normal | “silahkan berkas dambil di ruang saya” |


| Kategori | Contoh SMS |
| --- | --- |
| Promo | “Mau BONUS KEREN dari Indosat ooredoo? Ketik *345# dari HP kamu sekarang juga..!!” |
| Promo | “Selamat Paket 50 SMS ke semua operator Anda telah aktif. Cek kuota di *888#” |
| Penipuan | “Plg Yth,no Anda men-dpt hadiah dr: pengundiang PT. INDOSAT pin anda:b89c7h9 U/Melihat hadiah Anda klik: www.pemenangindosat- poin.webs.com” |
| Penipuan | “PESAN RESMI dri TRI CARE Selamat anda m-dptkan hadiah CEK TUNAI Sebesar Rp.45jt Pin undian 94AD653, U/Info kunjungi;www.undian- bimatri.blogspot.com” |

## 2. Jumlah Data

Berkas dataset awal berisi 1.143 SMS (Normal 569, Penipuan 335, Promo 239). Satu

SMS Normal ternyata duplikat dan dihapus saat pembersihan (keluaran program: duplikat

awal: 1), sehingga tersisa 1.142 SMS. Jumlah ini masih tergolong kecil dibandingkan sistem

filter SMS di dunia nyata, tetapi cukup masuk akal untuk proyek mahasiswa karena model bisa

dilatih dalam hitungan detik dan hasilnya tetap bisa diuji secara layak. Rincian jumlah tiap

kategori ada pada Tabel 3 dan Gambar 1, dan angka-angka ini diambil langsung dari keluaran

program.

*Tabel 3. Jumlah data per kategori*

| Kategori | Jumlah SMS | Persentase |
| --- | --- | --- |
| Normal | 568 | 49,74% |
| Promo | 239 | 20,93% |
| Penipuan | 335 | 29,33% |
| Total | 1.142 | 100% |


*Gambar 1. Diagram batang jumlah SMS per kategori (keluaran program)*

Penjumlahannya konsisten: 568 + 239 + 335 = 1.142. Terlihat bahwa datanya tidak

seimbang. SMS Normal hampir separuh dari seluruh data, sedangkan Promo paling sedikit.

Ketimpangan ini penting karena model cenderung lebih terbiasa dengan kategori yang

contohnya lebih banyak, dan dampaknya akan terlihat pada hasil pengujian di bagian 4.

Data kemudian dibagi 80% untuk melatih model dan 20% untuk menguji model.

Pembagian dilakukan secara stratified, artinya perbandingan antarkategori dijaga tetap sama di

kedua bagian. Hasilnya 913 data latih dan 229 data uji (keluaran program: latihan : 913 | uji:

229), seperti terlihat pada Tabel 4. Jumlah data uji per kategori sesuai laporan klasifikasi

program, dan jumlah data latih diperoleh dari total tiap kategori dikurangi data uji.

*Tabel 4. Pembagian data latih dan data uji*

| Kategori | Data latih | Data uji | Total |
| --- | --- | --- | --- |
| Normal | 454 | 114 | 568 |
| Promo | 191 | 48 | 239 |
| Penipuan | 268 | 67 | 335 |
| Total | 913 | 229 | 1.142 |

## 3. Model yang Digunakan

## 3.1 Pendekatan

Model yang dipakai termasuk machine learning dengan pendekatan supervised learning

untuk klasifikasi teks. Tugasnya adalah memilih satu dari tiga kategori untuk setiap SMS. Ada


dua model yang dicoba, yaitu Naive Bayes dan Logistic Regression. Keduanya dijalankan

dengan cara yang sama (pipeline TF-IDF dengan ngram_range=(1, 2)) agar perbandingannya

adil, dan Naive Bayes dipakai sebagai model utama, yaitu untuk tabel hasil uji, daftar kata

teratas, dan uji coba manual. Hasilnya juga sedikit lebih baik daripada Logistic Regression

(dibahas di bagian 4.6).

## 3.2 Mengubah Teks Menjadi Angka dengan TF-IDF

Komputer tidak bisa menghitung kata secara langsung, jadi teks perlu diubah menjadi

angka lebih dulu. Untuk itu dipakai TF-IDF. Idenya sederhana: kata yang sering muncul dalam

sebuah SMS diberi bobot, tetapi kata yang muncul di banyak SMS bobotnya dikecilkan karena

kurang membantu membedakan kategori. Sebaliknya, kata yang jarang muncul tetapi khas,

seperti “hadiah” atau “kuota”, mendapat bobot lebih besar.

TF-IDF di program ini diatur dengan ngram_range=(1, 2), sehingga yang dibaca bukan

hanya kata tunggal, tetapi juga pasangan dua kata berurutan. Pengaturan ini berguna karena

beberapa frasa lebih bermakna daripada kata terpisahnya. Contohnya “klik www”, yang

ternyata termasuk ciri kuat pada kategori Penipuan.

## 3.3 Naive Bayes

Naive Bayes (varian MultinomialNB) bekerja dengan menghitung peluang. Selama

pelatihan, model mencatat kata apa saja yang sering muncul pada tiap kategori. Ketika ada

SMS baru, model menghitung seberapa cocok kumpulan kata di SMS itu dengan kebiasaan

kata di masing-masing kategori, lalu memilih kategori dengan kecocokan tertinggi. Disebut

“naive” karena model menganggap setiap kata berdiri sendiri, padahal di kalimat nyata kata-

kata saling berkaitan. Anggapan yang terlalu sederhana ini memang tidak sepenuhnya benar,

tetapi dalam praktik klasifikasi teks sering sudah cukup baik.

Pertimbangan memakai Naive Bayes ada tiga. Pertama, modelnya cocok untuk data teks

yang sudah berbentuk bobot kata. Kedua, proses latihnya cepat dan ringan, jadi nyaman untuk

dataset berukuran ribuan data. Ketiga, hasilnya mudah dijelaskan, karena kita bisa melihat kata-

kata mana yang paling berpengaruh pada tiap kategori.

## 3.4 Logistic Regression sebagai Pembanding

Logistic Regression dipakai sebagai pembanding (dengan max_iter=1000). Model ini

mencari bobot untuk tiap kata, lalu menjumlahkannya untuk menentukan kategori yang paling

mungkin. Dengan adanya pembanding, pemilihan Naive Bayes tidak sekadar asumsi, tetapi

didukung hasil uji pada data yang sama.


## 3.5 Kata yang Dikenali Model

Setelah dilatih, kata-kata dengan nilai tertinggi pada tiap kategori dapat ditampilkan.

Program mengurutkan kata berdasarkan feature_log_prob_ model Naive Bayes, yaitu log

peluang kata pada tiap kategori, lalu mengambil 10 teratas. Hasilnya ada pada Tabel 5.

*Tabel 5. Sepuluh kata teratas tiap kategori (Naive Bayes)*

| Kategori | Kata dengan nilai tertinggi |
| --- | --- |
| Normal | aku, nama1, ada, yg, mau, aja, ya, di, nya, ga |
| Promo | di, paket, kuota, utk, sms, hari, tsel, bonus, promo, beli |
| Penipuan | anda, no, info, pin, www, hadiah, com, klik, klik www, ke |

Polanya masuk akal. Kategori Normal didominasi kata percakapan santai seperti “aku”,

“mau”, “aja”, dan “ga”. Kategori Promo berisi kata yang lekat dengan penawaran layanan

seluler, yaitu “paket”, “kuota”, “bonus”, “promo”, dan “beli”. Kategori Penipuan ditandai kata

seperti “hadiah”, “pin”, “klik”, dan “www”, yang sesuai dengan ciri pesan hadiah palsu dan

tautan mencurigakan. Kata “nama1” muncul di 76 SMS kategori Normal dan tidak ada di

kategori lain. Kata ini tampak seperti penanda untuk nama yang disamarkan di dalam data, jadi

tidak perlu dibaca sebagai kata biasa. Kata umum seperti “di” muncul di dua kategori karena

sangat sering dipakai, sehingga daftar ini menunjukkan kata dengan nilai tinggi pada tiap

kategori, bukan hanya kata yang eksklusif untuk kategori itu.

## 4. Tahapan

## 4.1 Mengumpulkan Dataset SMS

Tahap pertama adalah menyiapkan dataset, yaitu memuat berkas CSV

(dataset_sms_spam_v1) ke notebook Google Colab dan membacanya dengan library pandas.

Jika berkas belum ada di Colab, program meminta berkas diunggah lebih dulu. Setelah terbaca,

dicek apakah ada nilai kosong pada kolom teks dan label. Hasilnya nol untuk keduanya,

sehingga tidak ada baris yang harus dibuang karena kosong.

## 4.2 Memberikan Label pada Setiap Pesan

Setiap SMS pada dataset sudah memiliki label di kolom label, dengan angka 0 untuk

Normal, 1 untuk Penipuan, dan 2 untuk Promo. Program mengubah kolom ini menjadi bilangan

bulat agar bisa dipakai model. Angka-angka tersebut diterjemahkan kembali menjadi nama

kategori (normal, penipuan, promo) ketika hasil prediksi ditampilkan.


## 4.3 Preprocessing atau Pembersihan Teks

Pembersihan yang dilakukan di program cukup ringkas. Baris yang kolom teks atau

labelnya kosong dibuang lebih dulu (tidak ada yang terbuang pada data ini). Seluruh teks lalu

diubah menjadi huruf kecil dan spasi di awal maupun akhir dibuang, supaya kata “Anda” dan

“anda” dianggap sama. Setelah itu, SMS yang isinya kembar dihapus agar tidak ada pesan yang

sama dihitung dua kali. Pada data ini ditemukan satu duplikat awal, dan setelah seluruh proses

pembersihan tersisa 1.142 SMS. Perlu dicatat bahwa program tidak melakukan penghapusan

stopword maupun stemming, sehingga ejaan tidak baku tetap dibiarkan apa adanya. Selain itu,

TF-IDF secara bawaan memecah teks menjadi kata, mengabaikan tanda baca, dan tidak

menghitung huruf tunggal.

## 4.4 Membagi Data Latih dan Data Uji

Data dibagi menjadi 913 data latih (80%) dan 229 data uji (20%) menggunakan

pembagian acak dengan nilai random_state 42 agar hasilnya bisa diulang, serta stratify agar

proporsi kategori tetap terjaga. Data uji sengaja dipisahkan sejak awal dan tidak dipakai saat

pelatihan, sehingga bisa menjadi ukuran yang adil untuk menilai model pada SMS yang belum

pernah dilihat.

## 4.5 Melatih Model

Pada tahap ini TF-IDF dan model klasifikasi digabung dalam satu pipeline, sehingga

SMS mentah bisa langsung masuk dan diproses sampai menghasilkan prediksi. Pipeline Naive

Bayes dan pipeline Logistic Regression masing-masing dilatih memakai 913 data latih.

## 4.6 Menguji Model

Pengujian dilakukan dengan dua cara. Cara pertama adalah cross validation 5 lipatan

pada seluruh data, yaitu data dibagi menjadi lima bagian dan model diuji bergantian pada tiap

bagian. Cara kedua adalah menguji model pada 229 data uji yang dipisahkan tadi. Hasilnya ada

pada Tabel 6, sedangkan nilai tiap lipatan cross validation ada pada Tabel 7.

*Tabel 6. Perbandingan model*

| Model | CV rata-rata | Akurasi latih | Akurasi uji |
| --- | --- | --- | --- |
| Naive Bayes | 0,9072 | 0,9956 | 0,9214 |
| Logistic Regression | 0,9046 | 0,9967 | 0,9170 |


*Tabel 7. Akurasi cross validation tiap lipatan*

| Model | Lipatan 1 | Lipatan 2 | Lipatan 3 | Lipatan 4 | Lipatan 5 | Rata-rata |
| --- | --- | --- | --- | --- | --- | --- |
| Naive Bayes | 0,908 | 0,860 | 0,925 | 0,939 | 0,904 | 0,9072 |
| Logistic Regression | 0,891 | 0,834 | 0,943 | 0,943 | 0,912 | 0,9046 |

Kedua model menghasilkan akurasi uji di kisaran 92%, dengan Naive Bayes unggul tipis.

Selisihnya kecil sehingga tidak bisa dibilang satu model jauh lebih baik, tetapi hasil ini cukup

menjadi alasan memakai Naive Bayes sebagai model utama. Nilai cross validation (sekitar

0,90) juga mendekati akurasi uji, yang menandakan hasilnya tidak kebetulan baik pada satu

pembagian data saja. Meski begitu, nilai antarlipatan cukup bervariasi (Naive Bayes 0,860

sampai 0,939), sehingga angka akurasi sebaiknya dibaca sebagai kisaran, bukan nilai pasti.

Ada hal lain yang perlu diperhatikan, yaitu akurasi latih mendekati 100% sementara

akurasi uji sekitar 92%. Artinya model hampir hafal data latihnya dan sedikit menurun ketika

bertemu SMS baru. Rincian per kategori untuk Naive Bayes pada data uji ditunjukkan Tabel 8

dan Gambar 2, sedangkan hasil Logistic Regression ada pada Tabel 9 dan Gambar 3.

*Tabel 8. Hasil Naive Bayes per kategori pada data uji*

| Kategori | Precision | Recall | F1-score | Data uji |
| --- | --- | --- | --- | --- |
| Normal | 0,94 | 0,98 | 0,96 | 114 |
| Penipuan | 0,92 | 0,88 | 0,90 | 67 |
| Promo | 0,87 | 0,83 | 0,85 | 48 |
| Rata-rata makro | 0,91 | 0,90 | 0,90 | 229 |

*Gambar 2. Confusion matrix Naive Bayes pada data uji (keluaran program)*


*Tabel 9. Hasil Logistic Regression per kategori pada data uji*

| Kategori | Precision | Recall | F1-score | Data uji |
| --- | --- | --- | --- | --- |
| Normal | 0,94 | 0,98 | 0,96 | 114 |
| Penipuan | 0,91 | 0,88 | 0,89 | 67 |
| Promo | 0,87 | 0,81 | 0,84 | 48 |
| Rata-rata makro | 0,91 | 0,89 | 0,90 | 229 |

*Gambar 3. Confusion matrix Logistic Regression pada data uji (keluaran program)*

Precision menunjukkan seberapa tepat prediksi sebuah kategori, sedangkan recall

menunjukkan seberapa banyak SMS dari kategori itu yang berhasil dikenali. Kategori Normal

paling baik dikenali. Promo paling lemah (F1-score 0,85 pada Naive Bayes), sejalan dengan

jumlah datanya yang paling sedikit. Pada kategori Penipuan, recall 0,88 berarti sekitar 12%

SMS penipuan di data uji lolos dikira kategori lain, dan ini perlu diperhatikan karena kategori

inilah yang paling berbahaya bagi pengguna. Pola yang sama muncul pada Logistic Regression,

sehingga kelemahan ini tampaknya berasal dari data dan bukan dari pilihan model.

## 4.7 Menampilkan Hasil Klasifikasi Pesan

Sebagai contoh hasil pada data uji, 15 SMS pertama yang ditampilkan program (tabel

teks SMS, label asli, prediksi, dan status) seluruhnya diprediksi sesuai label aslinya. Contoh ini

hanya potongan kecil, sehingga ukuran kinerja yang sebenarnya tetap mengacu pada akurasi

0,9214 untuk seluruh 229 data uji.

Program juga dilengkapi uji coba manual, yaitu pengguna mengetik kalimat dan program

langsung menampilkan kategorinya; kalimat sebelum dimasukkan diubah ke huruf kecil, dan

pengujian dihentikan dengan mengetik “k”. Hasil dari empat kalimat yang dicoba tercantum


pada Tabel 10. Kalimat-kalimat ini ditulis sendiri dan tidak memiliki label asli, sehingga

hasilnya hanya gambaran perilaku model, bukan ukuran akurasi.

*Tabel 10. Hasil uji coba manual*

| Kalimat yang diketik | Hasil deteksi |
| --- | --- |
| selamat anda memenangkan undian 1 juta rupiah jangan lupa untuk klik link di bawah ini | Penipuan |
| syo buruan isi ulang diamond ep ep anda di henry store | Promo |
| selamat kamu memenangkan undian 1000 diamond ep ep | Penipuan |
| diamond ep ep kamu sudah habis ayo isi ulang lagi | Normal |

Kalimat pertama diprediksi Penipuan, sesuai dengan pola pesan hadiah palsu yang

meminta pengguna menekan tautan. Kata “anda” dan “klik” juga termasuk kata teratas kategori

Penipuan (Tabel 5). Kalimat kedua diprediksi Promo, masuk akal karena berupa ajakan isi

ulang. Kalimat ketiga diprediksi Penipuan, kemungkinan karena pola “selamat …

memenangkan undian” mirip pesan hadiah palsu; dugaan ini belum diuji secara khusus.

Kalimat keempat menarik untuk dicermati. Isinya berupa ajakan isi ulang yang terdengar

seperti promo, tetapi diprediksi Normal. Salah satu kemungkinan penyebabnya adalah gaya

bahasanya yang santai (“kamu”, “sudah”, “ayo”) mirip pesan percakapan, sementara kata khas

promo seperti “paket”, “kuota”, atau “bonus” tidak ada di kalimat tersebut. Kalimat kedua dan

keempat sama-sama membahas isi ulang diamond tetapi hasilnya berbeda, yang menunjukkan

model sangat bergantung pada kata-kata tertentu. Ini masih dugaan dan belum dibuktikan lewat

pengujian khusus.

## 4.8 Contoh Alur Pemrosesan Satu SMS

Agar alurnya lebih jelas, berikut satu SMS dari uji coba manual yang diikuti dari awal

sampai menjadi prediksi.

*Tabel 11. Alur satu SMS sampai menjadi prediksi*

| Langkah | Yang terjadi |
| --- | --- |
| 1. SMS masuk | “selamat anda memenangkan undian 1 juta rupiah jangan lupa untuk klik link di bawah ini” |
| 2. Pembersihan | Teks diubah menjadi huruf kecil dan spasi di tepi dibuang. Pada kalimat ini hasilnya sama karena sudah huruf kecil. |
| 3. TF-IDF | Kalimat dipecah menjadi kata dan pasangan dua kata, misalnya “selamat”, “anda”, “klik”, “selamat anda”, lalu tiap kata diberi bobot. Kata yang tidak pernah muncul di data latih diabaikan. |


| Langkah | Yang terjadi |
| --- | --- |
| 4. Naive Bayes | Model mencocokkan kata-kata tersebut dengan kebiasaan kata di tiap kategori. “anda” dan “klik” termasuk kata teratas Penipuan, sehingga menarik prediksi ke kategori itu. Setelah seluruh kata dihitung, kategori dengan kecocokan tertinggi dipilih. |
| 5. Hasil | Diprediksi Penipuan. |

## 5. Kesimpulan

Proyek ini bertujuan membuat sistem yang dapat membaca isi SMS dan menentukan

apakah pesan tersebut Normal, Promo, atau Penipuan. Dengan dataset 1.142 SMS dan model

Naive Bayes berbasis TF-IDF, sistem berhasil mencapai akurasi 0,9214 pada 229 data uji,

dengan rata-rata cross validation 0,9072. Logistic Regression sebagai pembanding

menghasilkan akurasi 0,9170, sehingga kedua model berada di tingkat yang hampir sama.

Manfaat sistem ini terutama ada pada kemampuannya membantu pengguna mengenali

SMS yang berpotensi berbahaya sebelum pesan itu ditanggapi. Ciri-ciri seperti kata “hadiah”,

“pin”, dan “klik www” mudah dikenali model, sehingga pesan hadiah palsu bisa ditandai lebih

cepat daripada jika diperiksa satu per satu secara manual.

Meski begitu, sistem ini punya beberapa keterbatasan. Pertama, model masih bisa salah

pada pesan yang kata-katanya ambigu, misalnya promo yang memakai bahasa seperti pesan

biasa atau pesan biasa yang memakai kata “anda” dan “paket”. Kedua, uji coba manual

menunjukkan model bisa memberi hasil yang berbeda pada kalimat bertema serupa, seperti

ajakan isi ulang diamond yang diprediksi Promo pada satu kalimat dan Normal pada kalimat

lain. Ketiga, kategori Promo memiliki data paling sedikit dan nilai F1-nya paling rendah,

sementara sekitar 12% SMS Penipuan pada data uji masih terlewat. Keempat, batas

antarkategori di dataset tidak selalu tegas. Ada pesan penawaran, seperti tawaran internet gratis

atau agen pulsa, yang berlabel Penipuan, dan ada notifikasi operator yang berlabel Normal,

sehingga model ikut mempelajari label yang kabur. Kelima, akurasi latih yang hampir 100%

dibanding akurasi uji sekitar 92% menunjukkan model cenderung terlalu menyesuaikan diri

dengan data latih. Terakhir, pengujian hanya memakai 229 SMS dari dataset yang sama, dan

sebagian pesan di dalamnya tampak berasal dari sekitar tahun 2013 sampai 2016, sehingga

hasilnya belum menggambarkan kinerja pada SMS terbaru.

Ke depan, kinerja mungkin bisa naik dengan menambah data Promo dan menyeragamkan

ejaan, tetapi dampaknya belum diuji di proyek ini.
