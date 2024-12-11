import { AstraDB } from "@datastax/astra-db-ts";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import 'dotenv/config';
import OpenAI from 'openai';
import { v4 as uuidv4 } from 'uuid';

// Create an OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const { ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_API_ENDPOINT, ASTRA_DB_NAMESPACE } = process.env;

// Initialize AstraDB client
const astraDb = new AstraDB(
  ASTRA_DB_APPLICATION_TOKEN!,
  ASTRA_DB_API_ENDPOINT!,
  ASTRA_DB_NAMESPACE!
);

const COLLECTION_NAME = 'chat_radiology'; // same collection as before

// Configure text splitting (chunking)
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

// Data you want to insert - you can load this from a JSON if needed.
// Each object represents a "document" you want to add to the RAG.
// You can customize `document_id`, `category`, and `text` as you wish.
const dataToInsert = [
  {
    document_id: "additional_document_1",
    category: "Sindrom Umum",
    text: `Dispepsia adalah kumpulan gejala atau sindrom gangguan saluran pencernaan atas. Dispepsia meliputi rasa nyeri, tidak nyaman, atau rasa terbakar di area gastroduodenum (epigastrium/ulu hati). Keluhan dapat disertai rasa perut penuh, cepat kenyang, mual, bahkan muntah. Etiologi dispepsia sering tidak diketahui, disebut dispepsia fungsional. Sedangkan dispepsia organik adalah kondisi yang diketahui penyebabnya.[2-6]

        Berdasarkan studi, 5 etiologi utama dispepsia adalah gastroesophageal reflux disease (GERD), obat-obatan, dispepsia fungsional, ulkus peptikum kronis, dan kanker lambung.[2]

        Tanda dan Gejala

        Karakteristik umum adalah rasa perut cepat penuh (kembung) setelah makan, cepat kenyang, atau istilah perut begah
        Keluhan area epigastrium atau ulu hati: rasa terbakar, tidak nyaman, bahkan nyeri
        Kadang disertai keluhan sering bersendawa, mual, muntah, dan nyeri dada bukan spesifik jantung[2-6]
        Dispepsia fungsional pada anak dan remaja didefinisikan oleh Komite Roma III sebagai setidaknya 2 bulan dengan keluhan:

        Nyeri atau ketidaknyamanan area epigastrium yang terus-menerus atau berulang
        Tidak ada bukti dispepsia berkurang dengan buang air besar (BAB), atau berhubungan dengan gangguan BAB seperti diare, konstipasi, maupun perubahan bentuk feses
        Tidak ada bukti kelainan pada proses inflamasi, anatomi, metabolik, atau neoplastik[1,2]
        Peringatan

        Terdapat beberapa red flags yang harus diperhatikan pada pasien dengan keluhan dispepsia, terutama gejala dan tanda yang mengarah kanker lambung. Tanda bahaya tersebut meliputi:

        Pasien berusia >55 tahun
        Perdarahan saluran cerna, termasuk hematemesis, melena, atau keduanya
        Cepat kenyang hingga timbul keluhan disfagia atau odinofagia
        Muntah berulang
        Penurunan berat badan sampai lebih dari 10% berat normal orang tersebut yang bukan karena upaya penurunan berat badan
        Limfadenopati
        Riwayat keluarga dengan kanker lambung atau esophagus

        Teraba massa di abdominal[2-6]
        Medikamentosa

        Terapi kasus dispepsia disesuaikan dengan gejala yang dirasakan pasien, terutama keluhan akibat peningkatan asam lambung. Pada setiap pasien, pilihan terapi bisa berbeda-beda. Perlu dipahami untung rugi penggunaan acid suppressant Untuk dispepsia.

        Pasien Dewasa

        Terapi yang paling sering digunakan untuk mengatasi gejala dispepsia fungsional antara lain antasida, H2 blocker, dan PPI (proton pump inhibitor). Beberapa studi menyebutkan pilihan terbaik untuk dispepsia fungsional pasien dewasa adalah golongan PPI. Perlu dipahami pedoman praktik untuk mengurangi utilisasi berlebih penghambat pompa proton. Pilihan PPI di antaranya:

        Omeprazole 2x20 mg/hari, atau

        Lansoprazole 2x30 mg/hari

        Antasida dapat diberikan dengan dosis 3 x 500-1000 mg/hari. Sedangkan H2 blocker  dapat dipilih salah satu dari obat di bawah ini:

        Ranitidin 2x150 mg/hari

        Famotidin 2x20 mg/hari

        Cimetidin 2x400-800 mg/hari
        Pasien dispepsia fungsional dengan gejala utama mual atau rasa penuh pada perut dapat membantu jika diberikan obat prokinetik untuk mempercepat pengosongan lambung. Berikut merupakan obat yang dapat dipilih:

        Domperidon 3x10 mg/hari
        Metoklopramid 3x10 mg/hari[4,6,7]
        Pasien Anak

        Kebanyakan kasus dispepsia pada anak dan remaja tidak memiliki penyakit yang serius. Jarang ditemukan infeksi Helicobacter pylori maupun ulkus peptikum. Umumnya pada anak ditemukan dispepsia fungsional yang akan membaik seiring waktu. Beberapa studi mendukung terapi non farmakologi untuk dispepsia fungsional pada anak, misalnya modifikasi diet, terapi perilaku kognitif untuk kasus kecemasan dan latihan relaksasi.[2,8]

        Sedangkan penelitian yang dipublikasikan terkait terapi farmakologi untuk disfungsi fungsional pada anak belum banyak. Pasien anak dengan keluhan dyspepsia rekuren atau lebih dari 2 minggu sebaiknya dikonsultasikan ke spesialis anak. Pilihan obat untuk dispepsia pada anak adalah golongan antasida (aluminium hidroksida atau magnesium hidroksida) dan prokinetik (metoclopramide atau domperidone).[8,9]

        Ketentuan dosis adalah sebagai berikut:

        Antasida: dosis 50-150 mg/kgBB per hari, dibagi menjadi 4 kali pemberian (setiap 6 jam). Satu tablet atau satu sendok takar obat 5 mL mengandung 200 mg aluminium hidroksida atau magnesium hidroksida, sehingga untuk anak >8 tahun dapat diberikan ½ atau 1 tablet atau sendok takar 3−4 kali sehari.
        Metoclopramide: 0,1-0,15 mg/kgBB, diberikan 3x sehari, maksimal 5 hari.

        Domperidone: anak dengan berat 15-35 kg dosis 0,25 mg/kgBB, diberikan 1-3x kali sehari, dosis maksimal 0,75 mg/kgBB. Sedangkan dosis anak dengan berat > 35 kg sama dengan dewasa, yaitu 10 mg diberikan 3x sehari.[2,8,9]

        Pilihan Terapi pada Kehamilan

        Pada ibu hamil terapi lini pertama yang dapat digunakan adalah antasida dan sukralfat sebagai pelindung mukosa lambung. Antasida yang menjadi pilihan adalah antasida yang tidak berbasis magnesium, karena dapat mengganggu kontraksi otot persalinan dan penyerapan zat besi. [3,5]

        Bila kondisi tidak kunjung membaik, penambahan obat golongan anti H2 seperti ranitidin dapat dipertimbangkan. Simetidin merupakan kontraindikasi karena adanya sifat antiandrogen.[2-7]

        Omeprazole termasuk ke dalam kategori C (FDA). Namun, berbagai studi selama 13 tahun memberikan hasil bahwa golongan PPI aman, baik pada periode konsepsi maupun selama kehamilan. [10]`
  },
  {
    document_id: "additional_document_2",
    category: "Penyakit Umum",
    text: `Common cold atau selesma adalah infeksi akut yang menyebabkan inflamasi cavum nasal, laring, trakea dan bronkus, dan merupakan salah satu kondisi yang paling umum ditemukan pada pasien rawat jalan. Sekitar 85% populasi akan mengalami common cold setidaknya satu kali dalam setahun.[1-3]

        Etiologi tersering adalah virus, di antaranya Rhinovirus sekitar 50% dan Coronavirus. Common cold pada umumnya dapat sembuh tanpa pengobatan, tetapi dapat mengurangi produktivitas sehingga memiliki dampak ekonomi. Selain itu juga dapat menjadi pemicu eksaserbasi gangguan pernapasan kronis, seperti penyakit paru obstruktif kronik (PPOK) dan asma.[1-3]

        Tanda dan Gejala

        Masa inkubasi selama 12-72 jam, dan penyakit dapat berlangsung selama 7-11 hari. Diversitas virus yang luas, komorbiditas, dan gangguan imun pasien menyebabkan manifestasi common cold sangat beragam.[3-5]

        Gejala common cold antara lain:

        Iritasi atau rasa kering pada hidung
        Sakit atau iritasi tenggorokan
        Nasal discharge, kongesti, bersin-bersin, dan batuk

        Sakit kepala dan malaise
        Penurunan indera penciuman dan perasa
        Demam jarang dirasakan dan umumnya ringan
        Diagnosis banding common cold penting dipertimbangkan untuk menyingkirkan kemungkinan penyakit lain yang serupa dan memastikan pemilihan medikamentosa yang tepat. Diagnosis banding common cold misalnya rhinitis alergi, faringitis, bronkitis akut, influenza, sinusitis, dan pertusis.[6,7]

        Peringatan

        Pasien dewasa dengan gejala common cold sebaiknya dirujuk ke fasilitas pelayanan kesehatan jika ditemukan:

        Nyeri dada, nyeri perut, atau sesak napas
        Gejala menetap lebih dari 10 hari
        Riwayat gangguan jantung atau penyakit pernapasan kronis
        Pasien lansia
        Nyeri telinga atau kepala berat, atau gejala-gejala lain yang mengarah ke komplikasi[8,9]
        Sementara itu, pasien anak harus  dirujuk jika ditemukan:

        Demam >38℃ selama 5 hari atau lebih, karena kecurigaan penyakit Kawasaki

        Nyeri dada, nyeri perut, atau sesak napas
        Bibir atau kuku kebiruan, atau kulit pucat dan teraba dingin
        Kejang atau penurunan kesadaran
        Anak usia <9 bulan
        Nyeri telinga atau kepala berat, atau gejala-gejala lain yang mengarah ke komplikasi
        Peringatan Medikamentosa

        Codein jangan diberikan untuk pasien common cold dengan batuk. Berdasarkan studi, codeine tidak bermanfaat untuk penanganan batuk. Obat ini lebih berisiko menimbulkan efek samping, bahkan dapat meningkatkan risiko kematian akibat depresi napas jika digunakan sebagai antitusif pada anak.[11]

        Kombinasi dekongestan-antihistamin-analgesik bisa meringankan gejala common cold pada orang dewasa dan anak usia >2 tahun. Kombinasi ini tidak disarankan pada anak usia <2 tahun. Akan tetapi, perlu diwaspadai efek samping akibat obat kombinasi ini, yaitu sedasi, rasa pusing, peningkatan asam lambung, mulut kering, serta mual.[12]

        Edukasi Pasien dan Orang Tua Pasien

        Pasien atau orang tua pasien perlu diberikan edukasi terkait common cold, di antaranya:

        Umumnya common cold akan sembuh sendiri dan pengobatan bertujuan untuk mengurangi gejala. Sebagian besar common cold disebabkan oleh virus sehingga tidak diperlukan pemberian antibiotik.[6,8]

        Cegah over konsumsi obat-obatan yang dijual bebas (over-the-counter / OTC), terutama untuk pasien anak di bawah usia 4 tahun.

        Beberapa suplementasi mineral, vitamin, dan fitofarmaka telah memiliki bukti ilmiah untuk meningkatkan sistem imun pasien dan mempercepat penyembuhan.

        Efektifitas preparat echinacea sebagai imunomodulator dalam penatalaksanaan common cold masih belum diketahui secara pasti.

        Terapi uap tidak bermanfaat sebagai penanganan common cold
        Medikamentosa

        Terapi common cold bertujuan untuk mengurangi keparahan dan durasi gejala. Tidak perlu diberikan antibiotik karena tidak efektif dan dapat meningkatkan risiko resistensi di masa depan. Obat-obatan yang dapat diberikan untuk pasien common cold adalah analgesik, dekongestan, antihistamin, mukolitik, antitusif, dan vitamin.[8,9]

        Tata Laksana Home Remedy

        Minum yang cukup, yaitu anak usia 7-12 bulan sekitar 800 mL; usia 1-3 tahun sekitar 1,3 L; usia 4-8 tahun sekitar 1,7 L; usia 9 tahun hingga dewasa sebanyak 2,1-2,4 L
        Minuman dapat berupa air putih, jus, kuah sup atau air lemon dengan madu. Hindari minuman beralkohol, kopi, atau soda berkafein
        Berkumur air garam dapat mengurangi nyeri tenggorokan, yaitu 1/4-1/2 sendok teh garam dilarutkan dalam 200 mL air. Untuk pasien anak, pastikan bahwa anak sudah dapat berkumur dengan benar tanpa tersedak
        Irigasi nasal dengan saline drop atau spray dapat mengurangi gejala pilek pada anak

        Campurkan madu dalam minuman untuk mengurangi batuk pada anak
        Dosis Pemberian Madu:

        Usia 12 bulan - 5 tahun: ½ sendok teh
        Usia 6-11 tahun: 1 sendok teh
        Usia 12-18 tahun: 2 sendok teh
        Diberikan 1 kali setiap 30 menit sebelum tidur
        Ukuran 1 sendok teh setara dengan 5 mL, tetapi dosis pemberian madu tidak harus tepat seperti obat[13‒15]
        Analgesik

        Analgesik diberikan jika ada keluhan demam atau nyeri. Dapat dipilih salah satu dari paracetamol atau ibuprofen.

        Dosis Paracetamol:

        Usia <12 tahun, yaitu 15 mg/kgBB per pemberian, maksimal 4 kali dalam sehari
        Usia >12 tahun: dosis 500  mg, diberikan 4 kali dalam sehari, dosis maksimal 3.250 mg per 24 jam
        Dewasa: 1.000 mg peroral, 4 kali sehari, dengan dosis maksimal 4 gram/hari
        Sediaan paracetamol berupa tablet 500 mg, sirup 120 mg/5 mL, dan drops 60 mg/0,6 mL[16]
        Dosis Ibuprofen:

        Usia <6 bulan: tidak dianjurkan
        Usia >6 bulan: dosis 10 mg/kgBB, diberikan 3 kali dalam sehari, dosis maksimal dalam 1 hari adalah 40 mg/kgBB
        Dewasa: 400 mg peroral, 4-6 kali sehari, dosis maksimal 3,2 gram/hari
        Ibuprofen tersedia dalam bentuk tablet 200 mg; kaplet 400 mg; serta sirup/suspensi 100 mg/5 mL dan 200 mg/ 5mL[17]
        Dekongestan

        Sebaiknya diberikan dekongestan topikal atau intranasal, karena memiliki potensi yang lebih baik daripada peroral atau sistemik. Namun, dekongestan topikal jangan digunakan dalam waktu lama agar mencegah rhinitis medikamentosa. Sementara, dekongestan peroral dapat dipilih efedrin atau pseudoefedrin.[6]

        Dosis Oxymetazoline Nasal:

        Di Indonesia, oxymetazoline topikal intranasal tersedia untuk dewasa (0,05% nasal spray 10 mL) dan untuk anak (0,025% nasal drops 10 mL). Dosis yang diberikan: 

        Dewasa: semprot hidung 0,05%, disemprotkan 1-2 kali ke masing-masing lubang hidung sebanyak 2-3 kali sehari jika perlu, durasi maksimal 5-7 hari berturut-turut
        Anak >6 tahun: semprot hidung 0,05%, disemprotkan 1-2 kali ke masing-masing lubang hidung sebanyak 2-3 kali sehari jika perlu, durasi maksimal 5-7 hari berturut-turut
        Anak 1-6 tahun: tetes hidung 0,025%, diteteskan 1-2 tetes ke setiap lubang hidung sebanyak 2-3 kali sehari jika perlu, durasi maksimal 5-7 hari berturut-turut[18]
        Dosis Pseudoefedrin:

        Dewasa dan anak >12 tahun: tablet konvensional diberikan 60 mg setiap 4-6 jam dengan dosis maksimal 240 mg/hari, sedangkan tablet lepas lambat diberikan 120 mg setiap 12 jam atau 240 mg setiap 24 jam

        Anak 6-11 tahun: 30 mg setiap 4-6 jam, dosis maksimal 120 mg/24 jam
        Tidak untuk anak usia <6 tahun [19]
        Di Indonesia, pseudoefedrin umumnya tersedia dalam bentuk kombinasi, di antaranya:

        Pseudoephedrine HCl 60 mg dan Triprolidine HCl 2,5 mg (contoh Tremenza®, Rhinofed®): dosis dewasa 1 tablet diberikan 3 kali/hari 
        Pseudoephedrine HCl 60 Mg dan Loratadine 5 Mg (contoh Rhinos®): dosis dewasa 1 tablet diberikan 3 kali/hari 
        Pseudoephedrine HCl 30 mg, Dextromethorphan 15 mg, dan Paracetamol 500 mg (contoh Panadol Cold & Flu®): dosis dewasa 1 tablet diberikan 3 kali/hari 
        Pseudoephedrine HCl 15 mg dan dextromethorphan 5 mg (contoh Triaminic® sirup): dosis anak 4-11 tahun 5 mL setiap 4-6 jam
        Dosis Efedrin:

        Dewasa dan anak usia >12 tahun: 60 mg, 3 kali/hari, di mana pasien lansia diberikan dosis awal 50%
        Anak usia 6-12 tahun: 30 mg, 3 kali/hari
        Tidak untuk anak usia <6 tahun[20]
        Bentuk obat kombinasi yang mengandung efedrin di antaranya:

        Ephedrine HCl 2,5 mg, Chlorpheniramine Maleate 1,3 mg, dan Paracetamol 135 mg per 5 mL (contoh OBH Nellco® sirup): dosis dewasa 15 mL, 4 kali/hari (jangan berikan resep dengan obat kombinasi lain yang juga mengandung paracetamol)
        Ephedrine HCl 12,5 mg, Chlorpheniramine Maleate 1 mg, Dextromethorphan 10 mg, dan Guaifenesin 100 mg (contoh Mixadin®): dosis dewasa 2 tablet, 3 kali/hari
        Antihistamin

        Efektivitas antihistamin dalam mengatasi rhinorrhea, baik generasi pertama dan kedua, tidak lebih efektif daripada plasebo. Antihistamin hanya diberikan jika common cold dikaitkan dengan reaksi alergi.[8]

        Dosis Cetirizine:

        Usia <2 tahun: efikasi dan keamanan cetirizine belum terbukti
        Usia 2-6 tahun: 2,5 mg/hari, dosis dapat ditingkatkan menjadi 5 mg/hari atau 2,5 mg/12 jam, dosis maksimal 5 mg /hari
        Usia >6 tahun-dewasa: 5-10 mg/hari
        Sediaan cetirizine berupa tablet 5 mg dan 10 mg, tablet salut 10 mg, kapsul 10 mg, sirup 5 mg/5 mL dan 10 mg/ 5 mL[21]
        Dosis Loratadine:

        Usia <2 tahun: efikasi dan keamanan loratadine belum terbukti
        Usia 2-6 tahun: 5 mg/hari
        Usia >6 tahun-dewasa: 10 mg/hari atau 5 mg/12 jam, dosis maksimal 40 mg/hari
        Sediaan loratadine berupa tablet 10 mg, tablet kunyah 5 mg, kapsul 10 mg, dan sirup 5 mg/5 mL[22]
        Mukolitik

        Mukolitik diberikan untuk meringankan gejala batuk berdahak dengan mengurangi kekentalan dahak sehingga dahak lebih mudah dikeluarkan.[8]

        Dosis Ambroxol:

        Ambroxol tablet 30 mg, pemberian dosis sebagai berikut:

        Anak usia <2 tahun: tidak direkomendasikan karena dikaitkan dengan sudden infant death

        Anak usia 2-5 tahun: dosis 7,5 mg-15 mg, 1 kali/hari
        Anak usia 6-12 tahun: dosis 15-30 mg, 1-2 kali/hari
        Anak usia >12 tahun-dewasa: dosis 30 mg, 2-3 kali/hari
        Ambroxol kapsul lepas lambat 75 mg, pemberian dosis:

        Dewasa: 75 mg/hari
        Ambroxol sirup 15 mg/5mL, pemberian dosis untuk anak:

        Usia < 2 tahun: tidak direkomendasikan karena dikaitkan dengan sudden infant death

        Usia 2-6 tahun: dosis 2,5 ml, 3 kali/hari
        Usia 6-12 tahun: dosis 5 ml, 2-3 kali/hari[23]
        Dosis Guaifenesin:

        Usia 6-12 tahun: 100-200 mg, setiap 4 jam, dengan dosis maksimal 1200 mg/hari
        Usia >12 tahun-dewasa: dosis 200-400 mg setiap 4 jam, atau dosis 600-1200 mg setiap 12 jam jika diberikan tablet lepas lambat, dosis maksimal 2400 mg per hari
        Guaifenesin tersedia dalam bentuk sediaan tablet 100 mg, 200 mg, dan 400 mg; sediaan sirup 100 mg/5 mL; serta sediaan tablet lepas lambat 600 mg dan 1200 mg[24]
        Dosis Bromhexine:

        Usia 2-5 tahun: 4 mg setiap 12 jam
        Usia >6 tahun-dewasa: 8 mg setiap 8 jam, dosis maksimal 96 mg/hari
        Sediaan bromhexine berupa tablet 8 mg dan sirup 4 mg/5 mL[25]
        Dosis N-asetilsistein:

        Usia 2-6 tahun: 100 mg, setiap 12-14 jam
        Usia >6 tahun-dewasa: 200 mg, setiap 8-12 jam
        Sediaan n-asetilsistein berupa kaplet 200 mg, sirup kering 100 mg / 5 ml, tablet effervescent 600 mg, dan granul larutan oral 200 mg[26]

        Antitusif

        Pilihan antitusif yang umum diberikan untuk kondisi common cold adalah dextromethorphan. Saat ini, di Indonesia hanya tersedia dextromethorphan bentuk kombinasi, misalnya:

        Dextromethorphan 10 mg, Chlorpheniramine Maleate 1 mg, Paracetamol 500 mg, Phenylpropanolamine HCl 15 mg (contoh Tuzalos®, Anadex®, Fludane®): dosis dewasa 1 tablet diberikan 3-4 kali/hari (jangan berikan resep dengan obat kombinasi lain yang juga mengandung paracetamol)
        Dextromethorphan 15 mg, Chlorpheniramine Maleate 1 mg, Guaifenesin 100 mg (contoh Konidin®, Komix®): dosis dewasa 1 tablet diberikan 3-4 kali/hari 
        Dextromethorphan 10 mg, Pseudoephedrine HCL 30 mg, Tripolidine HCL 1,25 g per 5 mL (contoh Actifed Plus Cough® sirup): dosis dewasa 5 mL diberikan 3-4 kali/hari 
        Vitamin

        Pemberian vitamin atau suplementasi dipercaya akan meningkatkan imunitas pasien. Beberapa suplemen yang dapat diberikan untuk pasien common cold adalah vitamin C dan Zinc.

        Vitamin C:

        Vitamin C sebagai imunomodulator dapat berasal dari buah maupun sayuran. Suplementasi asam askorbat dapat diberikan dengan dosis:

        Anak: 100 mg/hari, terbagi dalam 3 dosis terbagi
        Dewasa: 250 mg/hari, terbagi dalam 4 dosis[27]
        Zinc:

        Banyak penelitian mengenai penggunaan zinc untuk meringankan gejala common cold atau flu, baik dalam bentuk sediaan tablet, tablet hisap (lozenges), kapsul, dan sirup.

        Anak: dosis 10-30 mg/hari
        Dewasa: dosis 75 mg/hari
        Usahakan untuk mengonsumsi suplemen zinc dalam 1-2 hari sejak munculnya gejala
        Sediaan dalam bentuk sirup 10 mg/5 mL dan 20 mg/5 mL; serta tablet dispersible 20 mg[28]

        Pilihan Terapi pada Ibu Hamil dan Menyusui

        Batuk kering pada ibu hamil dan menyusui sedapat mungkin menggunakan tata laksana home remedy, untuk mencegah efek obat terhadap janin atau bayi.

        Antihistamin generasi kedua, cetirizine dan  loratadine, termasuk dalam FDA kategori B. Keduanya juga dilaporkan diekskresikan ke dalam ASI dalam jumlah minimal. Sehingga, penggunaannya pada kehamilan dan wanita menyusui menimbang aspek manfaat yang melebihi risiko.

        Obat analgesik dan antipiretik yang dapat diberikan pada ibu hamil hanya paracetamol, yang masuk ke dalam FDA kategori B. Sedangkan ibuprofen masuk dalam FDA kategori C.

        Obat mukolitik yang dapat diberikan pada ibu hamil adalah Bromhexine (FDA kategori A) dan N-asetilsistein (FDA kategori B). Namun, kedua obat ini belum diketahui ekskresinya ke dalam ASI sehingga sebaiknya tidak dikonsumsi oleh ibu menyusui.

        Ambroxol dan guaifenesin termasuk dalam FDA kategori C dan diekskresikan dalam jumlah minimal ke ASI, sehingga tidak direkomendasikan bagi ibu hamil dan menyusui.[23-27]`
  },
  {
    document_id: "additional_document_3",
    category: "Ginjal dan Saluran Kemih",
    text: `Nefrolitiasis atau batu ginjal atau renal calculi adalah batu yang terdapat di ginjal meskipun batu serupa dapat ditemukan juga di sepanjang traktus urinarius. Batu ginjal bisa asimptomatik, namun bisa juga memiliki manifestasi klinis signifikan seperti nyeri pinggang unilateral dan hematuria. Penyebab batu ginjal secara umum meliputi: Batu kalsium: umumnya berkaitan dengan hiperkalsiuria, hiperurikosuria, dan hipositraturia Batu asam urat Batu struvit akibat infeksi saluran kemih (ISK) Batu sistin Batu staghorn bilateral. Sumber: stockdevil, Freedigitalphotos, 2016. Batu staghorn bilateral. Sumber: stockdevil, Freedigitalphotos, 2016. Batu ginjal dapat turun ke saluran kemih di bawah ginjal, termasuk ureter dan uretra. Hal ini dapat menyebabkan kolik ginjal akut yang ditandai dengan nyeri hebat yang tiba-tiba, berasal dari panggul dan menjalar ke inferior dan anterior. Beberapa pasien juga mengeluhkan mual dan muntah. Selain kolik, pasien batu ginjal juga bisa mengeluhkan hematuria. Jika ukuran batu kecil dan tidak menghalangi saluran kemih, pasien bisa saja asimptomatik atau mengalami gejala ringan. CT scan abdominopelvis merupakan pencitraan pilihan untuk mengidentifikasi batu ginjal. CT scan abdominopelvis dapat mengetahui lokasi, diameter, dan densitas batu. Apabila tidak tersedia, USG merupakan alternatif yang baik. USG juga dapat mengidentifikasi hidronefrosis atau dilatasi uretra. Pada pasien yang mengalami kolik atau serangan akut akibat batu ginjal, tata laksana bersifat suportif. Hidrasi, analgesik seperti ketorolac, dan antiemetik seperti metoclopramide dapat diberikan untuk mengurangi keluhan. Memperbanyak minum air putih direkomendasikan untuk mencegah rekurensi batu ginjal. Eliminasi batu ginjal dapat dilakukan dengan tindakan invasif minimal seperti extracorporeal shockwave lithotripsy (ESWL), ataupun dengan pembedahan terbuka.[1-3]
    Pada prinsipnya, etiologi batu ginjal atau nefrolitiasis adalah ketidakseimbangan kimiawi antara zat-zat kimia dalam urine dengan air sebagai pelarutnya. Penyebab spesifik batu ginjal tergantung pada jenis batu ginjal itu sendiri, sebagai contoh batu kalsium dapat disebabkan oleh hiperkalsiuria, hiperurikosuria, dan hipositraturia Batu Kalsium 75% batu ginjal adalah batu kalsium. Batu kalsium dapat disebabkan karena hiperparatiroid, peningkatan penyerapan kalsium di usus, hiperurikosuria, hiperoksaluria, hipositraturia, ataupun hipomagnesuria. Kehamilan juga dapat meningkatkan risiko pembentukan batu kalsium.[8,10] Batu Struvit Batu struvit umumnya terbentuk akibat infeksi saluran kemih berulang oleh bakteri yang memiliki enzim urease, antara lain: Klebsiella sp, termasuk pneumoniae yang sudah ditemukan varian hipervirulen yang menyebabkan infeksi saluran kemih komplikata Proteus sp. Pseudomonas sp. Citrobacter Coagulase-negative Staphylococcus sp.[8,11] Batu Asam Urat Batu asam urat terjadi pada pasien dengan hiperurikosuria, misalnya pasien hiperurisemia dan gout.[1-4] Batu Sistin Batu sistin  berkaitan dengan kelainan genetik dimana terjadi defek pada fungsi metabolik sehingga terjadi gangguan reabsorpsi senyawa sistin, ornitin, lisin dan arginin di ginjal.[12] Obat Beberapa medikamentosa dapat meningkatkan risiko batu ginjal, seperti indinavir, atazanavir, dan guaifenesin.[1-4] Faktor Risiko Faktor berikut meningkatkan risiko batu ginjal: Gangguan kimia darah: hiperkalsiuria, sistinuria karena penyakit genetik, hiperoksaluria, hiperurikosuria, hipernatriuresis, dan hipositraturia Adanya komorbiditas: gangguan usus yang menyebabkan malabsorpsi seperti kolitis dan diare kronik. Arthritis gout meningkatkan risiko batu ginjal karena peningkatan asam urat darah. Komorbid lain adalah hiperparatiroid, medullary sponge kidney, renal tubular acidosis, infeksi saluran kemih, dan sindrom metabolik Diet, seperti diet tinggi kalsium Riwayat pribadi atau keluarga dengan batu saluran kemih[1-4,8] Gaya Hidup Penderita batu ginjal banyak ditemukan pada orang-orang yang kurang asupan cairan, buah-buahan dan sayuran. Penyakit ini juga banyak ditemui pada orang-orang yang berlebihan mengonsumsi garam, protein hewani, makanan tinggi purin, sumber oksalat, minuman bersoda, serta pengguna suplemen. Gaya hidup instan pekerja sibuk, misalnya lebih banyak duduk di depan gadget dan sering mengonsumsi makanan cepat saji, juga mendukung pembentukan batu ginjal.[8] Kehamilan Pada kehamilan, plasenta akan mensekresikan 1,25 – dyhydroxycholecalciferol, yang akan meningkatkan ekskresi dan penampungan oksalat, asam urat, natrium, dan kalsium dalam urine. Selain itu, uterus yang semakin meregang membesar selama kehamilan menggeser semua organ dalam tubuh di sekitarnya, termasuk ginjal dan ureter. Sekresi hormon progesteron yang meningkat selama kehamilan juga meningkatkan relaksasi otot polos uterus yang membesar, sehingga aliran urine pun dapat terhambat hingga bisa diam (terjadi urinary stasis). Diamnya aliran urine dan meningkatnya zat kimia yang dapat mencapai titik jenuh dan mengendap dalam urine mendukung terbentuknya batu ginjal.[10,13] Pekerjaan Pekerjaan risiko tinggi yang melibatkan paparan panas berlebih, seperti pekerja bangunan atau peleburan baja, dapat mengakibatkan kurangnya hidrasi. Tukang las dan pekerja pengecatan menggunakan spray juga terpapar kadmium dan asam oksalat yang bersifat nefrotoksik. Masalah ini meningkatkan kepekatan urine sehingga meningkatkan risiko terbentuknya batu ginjal.[8] Tempat Tinggal Tinggal di lingkungan urban memberi paparan suhu yang lebih hangat karena kurangnya vegetasi dan arsitektur bangunan urban. Hal ini meningkatkan pengeluaran air lewat keringat, dehidrasi, kepekatan urine, hingga terbentuk batu ginjal.[8] Penggunaan Obat-obatan Penggunaan suplemen kalsium, vitamin D, dan vitamin C diketahui justru meningkatkan insidensi batu saluran kemih. Riwayat penggunaan obat seperti probenecid, topiramate,  dan acetazolamide dapat mengganggu keseimbangan analit urine. Obat lain dapat meningkatkan risiko batu ginjal karena menimbulkan presipitasi langsung dalam urine, misalnya ciprofloxacin atau triamterene. Penggunaan antibiotik berlebihan juga dapat mengurangi jumlah koloni bakteri Oxalobacter formigenes, yang merupakan bakteri pelindung terhadap pembentukan batu ginjal.[8]
    Diagnosis batu ginjal atau nefrolitiasis perlu dicurigai pada pasien yang mengalami kolik ginjal akut, yang dapat disertai mual, muntah, dan hematuria. Meski demikian, beberapa pasien dengan batu ukuran kecil dapat tidak mengeluhkan gejala apapun. Diagnosis batu ginjal dapat dikonfirmasi dengan pencitraan CT scan abdominopelvis yang bermanfaat dalam menentukan lokasi, diameter, dan densitas batu. USG ginjal dapat menjadi alternatif, serta sekaligus dapat mengidentifikasi adanya hidronefrosis atau dilatasi uretra.[21] Anamnesis Pasien batu ginjal dapat datang dengan keluhan yang bervariasi, mulai dari tanpa keluhan, nyeri pinggang ringan hingga berat (kolik), nyeri saat berkemih (disuria), kencing berdarah (hematuria), sulit kencing (retensi urine), hingga tidak bisa kencing (anuria). Keluhan dapat disertai demam ataupun tanda-tanda gagal ginjal. Separuh pasien batu ginjal yang mengalami kolik akan mengeluhkan mual atau muntah juga. Kolik Renal Batu ginjal dapat turun ke saluran kemih di bawah ginjal, termasuk ureter dan uretra. Hal ini dapat menyebabkan kolik ginjal akut yang ditandai dengan nyeri hebat yang tiba-tiba, berasal dari panggul dan menjalar ke inferior dan anterior. Nyeri kolik ini terjadi tiga fase, yaitu: Fase akut dengan ciri serangan mendadak dan akut. Nyeri kolik yang timbul bersifat terus menerus, terasa sangat parah dan menyiksa. Nyeri bertambah hingga intensitas nyeri maksimal bisa dicapai dalam 30 menit hingga 6 jam setelah serangan Fase konstan. Setelah intensitas mencapai maksimal, nyeri akan menetap sampai diobati atau hilang. Durasi pada fase konstan pada umumnya 1–4 jam, dalam beberapa kasus dapat timbul sampai lebih dari 12 jam Fase penurunan nyeri. Nyeri berangsur menghilang. Durasi pada umumnya 1,5 hingga 3 jam Gejala penyerta yang dapat timbul antara lain mual, muntah dan nyeri perut. Gejala ini dapat timbul karena inervasi nervus celiacus dengan inervasi ginjal, yaitu inervasi ke perut dan usus. Kolik renal yang mereda bukan serta merta menandakan batu sudah keluar. Spontaneous passage harus dikonfirmasi dengan pemeriksaan penunjang, meskipun pasien melihat sendiri batunya keluar saat berkemih. Faktor Risiko Selain keluhan yang dialami saat itu, dokter juga perlu mengidentifikasi kemungkinan faktor penyebab. Tanyakan riwayat batu saluran kemih sebelumnya, riwayat penggunaan obat yang mungkin mengganggu keseimbangan urine, serta adanya komorbiditas. Riwayat hiperparatiroid, hiperurisemia, dan infeksi saluran kemih berulang akan meningkatkan risiko batu ginjal. Konsumsi suplemen, seperti suplemen kalsium, vitamin D, dan vitamin C, juga meningkatkan risiko batu ginjal. Tanyakan pula konsumsi obat yang dapat mengganggu keseimbangan urine, seperti probenecid, topiramate, dan acetazolamide. Gaya Hidup Pada anamnesis, tanyakan pola makan, asupan kalsium, asupan cairan, konsumsi air putih, garam, dan protein hewani. Investigasi besarnya asupan makanan sumber oksalat, purin, dan zat lain yang menjadi faktor risiko batu ginjal. Batu ginjal banyak dialami oleh individu dengan konsumsi garam, protein hewani, makanan tinggi purin, sumber oksalat, dan minuman bersoda berlebih.[1-3] Pemeriksaan Fisik Apabila pasien sedang mengalami kolik ginjal akut, pasien akan tampak sangat nyeri dengan onset tiba-tiba yang berasal dari panggul dan menjalar ke inferior dan anterior. Rasa sakit biasanya berhubungan dengan hematuria mikroskopis, mual, dan muntah. Nyeri dapat berpindah ke kuadran abdomen atas atau bawah apabila batu bermigrasi ke distal. Nyeri ketuk sudut costovertebra dapat ditemukan. Temuan pemeriksaan lain umumnya tidak khas. Nyeri testis dapat ditemukan, meskipun pada inspeksi tampak normal. Nyeri abdomen dapat ditemukan, tetapi tanda peritonitis tidak ditemukan yang membedakan nyeri abdomen karena kolik renal dengan nyeri karena organ intraperitoneum.[1-3] Diagnosis Banding Batu ginjal perlu dibedakan dari pyelonephritis, kehamilan ektopik, appendicitis akut, torsio testis dan kista ovarium. Appendicitis Akut Pada appendicitis akut, nyeri umumnya pada perut kanan bawah. Urinalisis akan normal dan tidak ditemukan batu ginjal pada CT scan ataupun USG. Kehamilan Ektopik Pada wanita usia subur, batu ginjal perlu dibedakan dengan kehamilan ektopik. Untuk membedakan dengan batu ginjal, akan didapat tes kehamilan positif dan nyeri goyang serviks. Pyelonephritis Pada pyelonephritis, akan ada tanda dan gejala infeksi saluran kemih dan nyeri pada sudut costovertebra. Pyelonephritis juga bisa terjadi bersamaan dengan batu ginjal apabila batu menyebabkan obstruksi ginjal. Torsio Testis Pasien dengan torsio testis bisa mengalami nyeri perut bawah dan testis yang juga bisa dialami pasien dengan batu ginjal. USG testis dapat membedakan keduanya. Kista Ovarium Pasien dengan kista ovarium juga dapat mengeluhkan nyeri pada pinggang atau perut bawah. Untuk membedakan dengan batu ginjal, dilakukan USG dimana akan tampak kista pada adneksa.[1-3] Pemeriksaan Penunjang Pemeriksaan penunjang pada batu ginjal bertujuan untuk mengonfirmasi diagnosis atau menyingkirkan diagnosis banding. CT Scan dan USG dapat mengonfirmasi diagnosis batu ginjal dan mengevaluasi besar serta lokasi batu. CT Scan CT scan tanpa kontras adalah pemeriksaan pencitraan awal yang disarankan untuk mengevaluasi batu ginjal. Pemeriksaan ini merupakan prosedur pilihan kecuali pada pasien yang hamil karena paparan radiasi yang besar dari CT Scan berpotensi teratogenik pada janin. CT scan abdominopelvis dapat mengidentifikasi lokasi, diameter, dan densitas batu.[1,2,8,10] USG USG sendiri lebih disukai digunakan sebagai prosedur pencitraan pertama di Indonesia. Hal ini karena ketersediaannya yang lebih luas dibandingkan CT Scan. Meski demikian, tingkat akurasi USG sangat bergantung dengan keahlian pemeriksa. Prosedur ini dinilai aman, tidak ada risiko paparan radiasi, dapat diulang, dan harganya lebih terjangkau dibandingkan CT Scan. USG dapat mengidentifikasi batu pada kaliks ginjal, pelvis ginjal, dan pyeloureteric and vesico-ureteral junctions. USG juga dapat mengevaluasi dilatasi saluran kemih atas.[2] Foto Polos Abdomen Foto polos abdomen dapat memperlihatkan gambaran opak pada batu radioopak seperti batu kalsium. Batu jenis lain, seperti asam urat dan sistin, tidak tampak pada modalitas pencitraan ini.[1-3] Intravenous Pyelography Intravenous Pyelography (IVP) dapat memberikan informasi anatomis dan fungsional, tetapi sudah jarang digunakan bila memungkinkan dilakukan CT-Scan.[1-3] Urinalisis Pada urinalisis, komponen yang diperiksa meliputi kalsium urine, pH dengan dipstick, dan analisis mikroskopik. pH urine rendah (di bawah 5,5) dapat terjadi pada batu ginjal akibat asam urat. pH tinggi (di atas 7) dapat terjadi pada batu infeksi. Keberadaan bakteriuria atau pyuria  dapat menandakan batu ginjal oleh infeksi. Urinalisis dapat dilakukan dengan pengumpulan urine 24 jam, urine 12 jam dari saat malam hari, maupun urine sewaktu dan rasio kalsium-kreatinin sewaktu. Untuk mendeteksi oksalat dan seberapa besar risiko pembentukan batu oksalat, dapat dilakukan analisis oksalat urine 24 jam.[1,2,4,8] Kultur Urine Kultur urine jarang diperlukan. Kultur urine dapat digunakan pada kasus-kasus yang tidak respon dengan pengobatan adekuat atau pada kasus yang dicurigai adanya ko-infeksi. Laboratorium Kimia Darah Pemeriksaan laboratorium kimia darah meliputi kadar kalsium, kreatinin, kalium, bikarbonat, dan asam urat. Nilai abnormal menandakan adanya penyakit yang dapat menyebabkan batu ginjal, seperti kadar kalsium serum yang tinggi pada hiperparatiroidisme. Pemeriksaan serum intact parathyroid hormone diindikasikan jika diduga ada kelainan paratiroid primer.[8] Pencitraan pada Kehamilan Pada pasien hamil, modalitas pencitraan yang dipilih sebagai lini pertama adalah USG ginjal. Dilatasi ureter bagian distal dapat menjadi pembeda antara dilatasi akibat obstruksi pada batu ginjal dengan dilatasi ureter fisiologis pada kehamilan. MRI, terutama MRU (magnetic resonance urography), direkomendasikan menjadi lini kedua pencitraan batu ginjal pada ibu hamil karena risiko paparan radiasi yang lebih kecil ketimbang CT scan. MRI atau MRU pada pasien batu ginjal yang hamil boleh dikerjakan dengan atau tanpa kontras gadolinium, mengingat gadolinium terbukti teratogenik pada pengujian pada hewan tetapi tidak pada manusia. American College of Obstetricians and Gynecologists (ACOG) masih merekomendasikan penggunaan gadolinium sebagai kontras untuk MRI pada pasien batu ginjal yang hamil jika keuntungan dianggap lebih besar daripada risiko.[9,10] Klasifikasi Batu ginjal, sebagaimana batu saluran kemih lainnya, dapat diklasifikasikan berdasarkan ukuran, lokasi, karakteristik pencitraan sinar X, etiologi terbentuknya batu, komposisi batu, dan risiko kekambuhan.[1,2] Ukuran Batu ginjal biasanya diklasifikasikan dalam 1 atau 2 dimensi. Batu ginjal secara umum dibagi dalam beberapa ukuran, yakni 5 mm, 5-10 mm, 10-20 mm, dan lebih besar dari 20 mm.[1,2] Lokasi Batu ginjal dapat terletak di kaliks ginjal superior, kaliks ginjal medial, kaliks ginjal inferior, pelvis renalis, ureter proksimal atau distal, dan buli (kandung kemih).[1,2] Karakteristik Pencitraan Sinar X Pada foto polos, batu ginjal dapat bersifat radioopak atau radiolusen. Contoh batu radioopak adalah kalsium oksalat dan batu kalsium fosfat. Contoh batu radiolusen adalah batu asam urat, amonium urat, dan batu obat-obatan. Contoh batu dengan opasitas rendah adalah batu apatit dan sistin.[1,2] Tabel 1. Jenis Batu Ginjal Berdasarkan Karakteristik Pencitraan Sinar X Radioopak	Opasitas Rendah	Radiolusen Kalsium Oksalat	Magnesium Amonium Fosfat	Asam urat Kalsium Fosfat	Apatit	Amonium urat Sistin	Xantin Obat-obatan Sumber: dr. Eveline, Alomedika, 2022.[1,2] Komposisi Batu Batu ginjal dapat digolongkan menjadi batu kalsium (batu kalsium oksalat dan batu kalsium fosfat), batu magnesium ammonium fosfat (struvit), batu apatit, batu sistin, batu asam urat, batu ammonium urat, batu xantin, dan batu obat-obatan.[1,2]
    Tidak semua batu ginjal memerlukan penatalaksanaan. Batu yang berukuran kecil dapat keluar sendiri saat pasien berkemih. Pada saat pasien sedang mengalami kolik ginjal akut, maka penatalaksanaan bertujuan meredakan nyeri dan membuat pasien nyaman, baru kemudian mengeluarkan batu. Batu yang tidak keluar secara spontan dapat ditata laksana dengan intervensi medis seperti litotripsi. Penanganan Kolik Renal Bila pasien datang dengan kolik renal, dapat dilakukan pemasangan akses intravena untuk hidrasi dan pemberian obat-obatan intravena. Bila tidak ada obstruksi atau infeksi, dapat diberikan analgesik, antiemetik, dan antidiuretik. Analgesik Karena nyeri kolik umumnya cukup berat, analgesik yang digunakan dapat berupa: Ketorolac dengan dosis awal 30–60 mg intramuskular (IM) atau 30 mg intravena (IV), diikuti 30 mg setiap 6–8 jam sesuai kebutuhan. Pada pasien berusia di atas 65 tahun, dosis diturunkan setengahnya. Sediaan intranasal juga sudah diteliti efikasinya Morfin dengan dosis 10 mg setiap 4 jam sesuai kebutuhan. Awasi potensi efek samping depresi napas, sedasi, konstipasi, potensi adiksi, mual, dan muntah Meperidin (60–80 mg meperidin ekuivalen dengan 10 mg morfin). Dosis 50–150 mg IM setiap 3–4 jam Pilihan terapi oral adalah obat antiinflamasi nonsteroid (OAINS) seperti natrium diklofenak 100-150 mg/hari selama 3-10 hari. Antiemetik Pasien dengan kolik renal umumnya mengalami mual dan muntah. Obat yang dapat diberikan adalah metoclopramide dosis 10 mg IV atau IM setiap 4–6 jam sesuai kebutuhan. Antidiuretik Desmopresin (DDAVP) dapat menurunkan nyeri kolik renal. Bila diberikan melalui semprotan nasal, maka dosis yang digunakan adalah 40 mcg. Bila diberikan secara IV, maka dosis yang digunakan adalah 4 mcg. Antibiotik Antibiotik hanya diberikan apabila ada potensi infeksi saluran kemih seperti adanya piuria, bakteriuria, demam, atau leukositosis dengan penyebab lain disingkirkan.[1-3] Terapi Konservatif Batu berukuran ≤ 0,4 cm memiliki kemungkinan 95% untuk keluar sendiri dalam 40 hari. Pasien dengan batu terletak di distal dengan ukuran < 10 mm boleh dicoba tata laksana konservatif jika nyeri terkendali, serta tidak ada bukti menurunnya fungsi ginjal atau terjadinya sepsis.[1-3] Modifikasi Gaya Hidup Modifikasi gaya hidup tanpa intervensi medis lain dapat dianjurkan bagi pasien dengan batu berukuran ≤ 4 mm maupun kasus batu ginjal berulang. Jenis dan isi gaya hidup yang dikendalikan disesuaikan dengan jenis batu. Tetapi untuk semua jenis batu, pasien dapat dianjurkan untuk: Cukup hidrasi: Pasien dapat disarankan memperbanyak minum air putih minimal 2,5–3 liter sehari (minimal 8 gelas per hari) atau hingga urine berwarna jernih. Meningkatkan aktivitas fisik: Pasien juga disarankan menjalani aktivitas fisik ringan-sedang 150 menit per minggu.[1-3] Modifikasi Diet Minta pasien menghindari minuman bersoda, mengurangi konsumsi kopi dan teh, dan menjalani diet rendah garam. Sampaikan untuk menjaga asupan kalsium dalam batas normal (700-1000 mg). Konsumsi kalsium bertujuan mencegah penarikan kalsium dari tulang ke dalam darah, yang dapat meningkatkan kadar kalsium urine setelah difiltrasi oleh ginjal. Minta pasien menghindari asupan protein berlebih dari makanan. Konsumsi protein 0,8 g/kg/hari. Batasi protein hewani dan hindari suplemen whey protein berlebihan. Protein dipecah menjadi purin hingga asam urat, sehingga protein berlebih dapat meningkatkan asam urat dalam urine, menimbulkan hiperurikosuria yang mendukung pengendapan dan pembentukan batu.[3,8] Rujukan Pasien yang memenuhi indikasi berikut perlu segera dirujuk ke dokter spesialis urologi untuk dilakukan intervensi lanjutan. Kegagalan Terapi Konservatif Terapi konservatif dinyatakan gagal bila nyeri memburuk, muncul tanda-tanda sepsis, atau output urine menurun.[3] Indikasi Rujuk pada Penderita Asimtomatik Indikasi rujuk untuk intervensi oleh spesialis urologi pada pasien asimptomatik adalah: Ukuran batu ≥ 5 mm Pasien hanya memiliki 1 ginjal Terjadi obstruksi kronis Terjadi infeksi saluran kemih berulang Batu ginjal yang tidak bergejala perlu dimonitor ketat dengan follow up klinis dan pencitraan setiap tahunnya. Intervensi perlu dipertimbangkan dan dilakukan setelah 2-3 tahun, atau lebih cepat bila batu cepat membesar melebihi 5 mm, menyebabkan obstruksi, infeksi, atau nyeri.[3] Medikamentosa Batu kalsium dapat dicegah dengan pemberian diuretik thiazide, seperti hydrochlorothiazide (HCT), yang dapat mengurangi ekskresi kalsium dalam urine. Kalium sitrat juga dapat digunakan untuk meningkatkan kadar sitrat urine. Untuk batu struvit, dapat dilakukan terapi antibiotik sesuai bakteri penyebab. Sedangkan untuk batu asam urat, pengendalian kadar asam urat dapat dilakukan dengan diet dan medikamentosa seperti alopurinol dan kolkisin.[8] Untuk batu oksalat, saat ini sedang berjalan penelitian beberapa agen biologik untuk mengendalikan hiperoksaluria, seperti SLC26 inhibitor dan bakteri. Prinsip utamanya mengurangi penyerapan oksalat di usus sehingga dapat mengurangi jumlah oksalat yang masuk ke urine dan dapat berkumpul membentuk batu.[16] Terapi Ekspulsif Obat α-blocker seperti tamsulosin dengan dosis 400 mcg sekali sehari (indikasi off-label) selama 1 bulan mungkin mempercepat pengeluaran batu distal secara spontan. Obat ini dapat menimbulkan efek samping berupa hipotensi postural dan ejakulasi retrograde. Pilihan lain dapat diambil dari golongan calcium-channel inhibitor seperti nifedipine.[1-3] Pembedahan Pilihan pembedahan bergantung pada preferensi pasien, lokasi dan ukuran batu, jumlah dan kompleksitas batu, komorbiditas yang diderita, dan apakah pasien hanya memiliki 1 ginjal. Pilihan prosedur yang dapat dilakukan antara lain ureteroskopi (URS), extracorporeal shockwave lithotripsy (ESWL), nefrolitotomi perkutan (PCNL), retrograde intrarenal surgery (RIRS), atau endoscopic combined intrarenal surgery (ECIRS). Untuk tata laksana bedah, pasien dirujuk ke dokter spesialis urologi.[1-3] Ureteroskopi (URS) / Retrograde Intrarenal Surgery (RIRS) Menurut European Association of Urology Urolithiasis Guidelines, ureteroskopi atau RIRS direkomendasikan sebagai terapi lini pertama untuk batu ginjal berukuran < 2cm. Ureteroskopi merupakan prosedur yang paling biasa dikerjakan dan angka bebas batunya tinggi. Prosedur ini menggunakan endoskopi transuretral, yang memungkinkan untuk mengangkat dan menghancurkan batu dengan teknik laser. Biasanya pasien dipasang stent setelah prosedur untuk memberi ruang pemulihan inflamasi, yang kemudian dikeluarkan 1-2 minggu pasca operasi.[1-3] Extracorporeal Shockwave Lithotripsy Indikasi ESWL adalah batu berukuran 0,5 cm sampai 2 cm. Prosedur ini menggunakan ultrasonic shockwave dari luar tubuh pasien yang ditembakkan ke arah mana terletak batu ginjal pasien. Prosedur ini bertujuan menghancurkan batu menjadi fragmen dan bubuk kecil agar bisa keluar secara spontan dan tanpa nyeri lewat aliran urine. Prosedur ini perlu dikerjakan berulang sampai batu hancur sempurna, dan kadang fragmen kecil batu ginjal yang tersisa dapat menimbulkan kolik ginjal akut. Selain itu, ada potensi timbulnya komplikasi berupa steinstrasse, dimana batu yang telah dipecah di tahap awal justru membentuk kompleks yang dapat menyebabkan obstruksi ureter.[1-3] Nefrolitotomi Perkutan (PCNL) Ukuran batu di atas 2 cm dan formasi batu kompleks termasuk jenis staghorn calculi adalah indikasi untuk dilakukan pembedahan invasif. Menurut European Association of Urology Urolithiasis Guidelines, PCNL direkomendasikan sebagai terapi baku emas untuk batu ginjal berukuran ≥ 2cm. Prosedur ini menggunakan jarum dan guidewire untuk mengakses secara perkutan menuju ginjal. Morbiditasnya lebih tinggi dan biasanya hanya dikerjakan pada batu ginjal yang berukuran besar atau ketika metode lain telah gagal.[1-3] Kekurangan metode ini adalah perlunya pengerjaan berulang hingga batu hancur sempurna atau cukup kecil dan permukaannya aman untuk bisa dikeluarkan sendiri lewat aliran urine. Proses invasif ini memiliki potensi timbulnya abses perkutan karena masuknya bakteri kulit ke jaringan dalam. Risiko komplikasi lainnya berupa steinstrasse.[17] Endoscopic Combined Intrarenal Surgery (ECIRS) Pada prinsipnya, prosedur ini adalah flexible ureteroscopy (FUS) dan PCNL yang dikerjakan secara simultan. ECIRS dimungkinkan untuk dikerjakan seiring dengan berkembangnya peralatan litotripsi yang fleksibel dan tidak kaku, sehingga memungkinkan dikerjaannya FUS sebagai pilihan terapi tambahan setelah PCNL. Prosedur ini menghasilkan kehilangan darah dan durasi perawatan pasca operasi yang lebih sedikit. ECIRS dinilai cukup aman bagi penderita batu ginjal kompleks, namun studi lebih lanjut masih diperlukan.[17] Terapi Suportif Pencegahan komplikasi infeksi dan urosepsis hingga sepsis diupayakan dengan Mengobati infeksi saluran kemih sesegera mungkin dengan antibiotik empiris berspektrum luas atau sesuai hasil kultur urine. Jika ditemukan kasus batu ginjal atau batu saluran kemih lain yang bersifat obstruktif disertai tanda klinis dan tanda laboratorium infeksi di unit gawat darurat, segera drainase dengan nefrostomi dan berikan antibiotik sistemik. Berikan juga analgesik bila diperlukan, kemudian rujuk dengan dokter spesialis urologi. Berikan antibiotik profilaksis berdasarkan hasil pemeriksaan urine lengkap dan kultur urine atau antibiotik berspektrum luas sebelum prosedur operasi. Batasi durasi pengerjaan operasi untuk prosedur PCNL, ESWL, maupun ureteroskopi.[11,15] Penanganan Pasien Hamil Expectant management (tata laksana konservatif) menjadi lini pertama terapi batu ginjal pada pasien hamil. Tata laksana konservatif yang dilakukan adalah hidrasi, antiemetik, dan analgesia. Jika urinalisis atau presentasi awal mengindikasikan infeksi, observasi disertai dengan terapi antibiotik dapat dilakukan. Antibiotik yang dapat digunakan hanyalah yang aman bagi kehamilan, seperti amoxicillin. Penggunaan alpha-blocker seperti tamsulosin sebagai medical-expulsion therapy belum banyak diteliti keamanannya pada ibu hamil.[9,10] Indikasi intervensi pada dasarnya sama seperti pada pasien yang tidak hamil, dengan tambahan mempertimbangkan risiko komplikasi kehamilan. ACOG merekomendasikan prosedur operasi nonobstetrik pada trimester kedua kehamilan, untuk menghindari risiko mengganggu perkembangan fetus dan risiko kontraksi prematur. Prosedur dilaksanakan dengan panduan USG dan dosis anestesi sekecil mungkin. Ureteroskopi dilaporkan membutuhkan waktu pembiusan yang lebih singkat dan lebih sedikit menimbulkan risiko kontraksi prematur dan infeksi dibandingkan nefrostomi perkutan dan ureteric stent. PCNL tidak direkomendasikan karena membutuhkan posisi pasien pronasi dan dosis anestesi yang besar. SWL tidak direkomendasikan karena meningkatkan risiko gangguan pertumbuhan janin, malformasi kongenital, dan kematian janin dalam rahim.[9,10]
    Prognosis batu ginjal atau nefrolitiasis tergantung pada ukuran batu, lokasi, dan komplikasi yang dialami pasien. Batu yang berukuran kecil mungkin dapat keluar sendiri. Sementara itu, batu berukuran lebih besar dapat menyebabkan komplikasi berupa obstruksi saluran kemih hingga sepsis dan urosepsis.[1-3] Komplikasi Komplikasi batu ginjal atau nefrolitiasis dapat dibagi menjadi komplikasi batu itu sendiri serta komplikasi dari tindakan bedah. Komplikasi Batu Ginjal Batu ginjal menghambat aliran urine sehingga meningkatkan risiko hidronefrosis dan infeksi saluran kemih. Infeksi saluran kemih yang pembilasannya terhambat atau bahkan mengalami stasis oleh obstruksi akan menyebar ke darah dan organ lain sehingga dapat terjadi urosepsis atau sepsis. Urosepsis menjadi penyebab terbesar mortalitas terkait batu ginjal.[3] Batu ginjal juga merupakan faktor risiko terjadinya penyakit ginjal kronik. Karsinoma sel skuamosa pelvis ginjal sangat jarang terjadi, tetapi pernah dilaporkan pada pasien yang mengalami batu ginjal berulang. Pada ibu hamil, batu ginjal meningkatkan risiko kehamilan preterm, preeklampsia, dan indikasi untuk melahirkan dengan prosedur sectio caesarea (SC).[9,18] Komplikasi Tindakan Komplikasi terkait pemasangan stent meliputi hematuria, sering merasa ingin kencing tidak tertahankan, sering berkemih berkali-kali, nyeri saat berkemih, dan nyeri daerah flank. Pasien bisa diyakinkan bahwa ini tidak terlalu masalah kecuali gejala memberat, persisten, atau berkaitan dengan gejala sistemik, retensi urine, atau temuan kultur urine positif. Gejala-gejala kolik akibat stent dapat dicoba diredakan dengan obat-obatan jenis alpha blockers seperti tamsulosin. Prosedur operasi invasif juga meningkatkan risiko infeksi. Risiko ini meningkat seiring semakin lama waktu yang dibutuhkan untuk melangsungkan prosedur.[3] Prognosis Batu ginjal yang berukuran 4 mm atau kurang, terlebih jika letaknya di distal, memiliki kemungkinan yang tinggi untuk keluar dengan sendirinya tanpa memerlukan intervensi medis. Semakin besar suatu batu maka semakin tinggi kemungkinan terjadinya obstruksi serta semakin meningkat kebutuhan intervensi medis. Batu yang letaknya lebih distal memang lebih mungkin keluar secara spontan saat pasien berkemih. Tetapi jika ukuran batu besar, maka batu bisa menyumbat ureter ataupun uretra.[14,15] Beberapa pasien mungkin menganggap bahwa batu ginjalnya sudah keluar secara spontan jika kolik renal mereda. Namun, meredanya kolik renal bukan merupakan tanda pasti bahwa batu ginjal sudah keluar. Kecuali jika pasien melihat secara langsung batu ginjal keluar ketika buang air kecil, pasien mungkin mengalami silent ureteral stone.[22] Risiko komplikasi dan mortalitas meningkat dengan adanya komorbiditas seperti obesitas, diabetes, sindrom metabolik, dan infeksi saluran kemih berulang. yang tidak melihat secara langsung batu ginjal keluar ketika buang air kecil. Rekurensi Batu ginjal memiliki angka rekurensi yang tinggi. Faktor risiko rekurensi batu ginjal antara lain onset awal (usia kurang dari 25 tahun), nefrokalsinosis, dan batu ginjal yang berkaitan dengan penyakit dan komorbiditas seperti hiperparatiroid dan hiperurisemia. Risiko sepsis meningkat dengan semakin kompleksnya batu ginjal yang diderita.[3,8,10]
    Edukasi dan promosi kesehatan terkait penyakit batu ginjal yang utama adalah mengenai perubahan gaya hidup. Sampaikan pada pasien bahwa beberapa perubahan gaya hidup dapat bermanfaat mencegah timbulnya batu ginjal dan membantu mengeluarkan batu yang kecil secara spontan. Edukasi Pasien Sampaikan pada pasien bahwa batu yang berukuran 4 mm atau kurang memiliki kemungkinan yang tinggi untuk dikeluarkan secara spontan saat pasien berkemih. Tetapi bila batu berukuran lebih besar dan lokasinya pada saluran kemih atas, maka intervensi medis umumnya diperlukan. Sarankan pasien untuk memperbanyak minum air putih, minimal 2,5–3 liter sehari (setara 8 gelas per hari) atau hingga urine berwarna jernih. Selain itu, sampaikan bahwa aktivitas fisik ringan-sedang 150 menit per minggu dapat membantu meningkatkan status kesehatans ecara umum, mencegah terbentuknya batu ginjal, dan membantu mengeluarkan batu ginjal bila sudah terbentuk.[1-3] Minta pasien menghindari minuman bersoda, mengurangi konsumsi kopi dan teh, dan menjalani diet rendah garam. Sampaikan untuk menjaga asupan kalsium dalam batas normal (700-1000 mg) serta menghindari asupan protein berlebih dari makanan. Pada pasien dengan hiperurisemia, anjurkan diet rendah purin.[3,8] Sampaikan pada pasien bahwa kolik renal yang mereda bukanlah pertanda bahwa batu sudah keluar. Pasien mungkin saja mengalami silent ureteral stone. Jelaskan bahwa pasien sebaiknya bertanya kepada dokter mengenai hal tersebut bila merasa ragu agar dapat dilakukan konfirmasi.[22] Upaya Pencegahan dan Pengendalian Penyakit Hidrasi yang cukup dan aktivitas fisik rutin dapat mencegah terbentuknya batu ginjal. Buah-buahan yang mengandung asam sitrat dan asam malat dapat bermanfaat menjaga keseimbangan ion dalam urine.[8,19] Pembatasan konsumsi garam dapat mencegah batu ginjal. Konsumsi garam berlebih meningkatkan kadar natrium dalam darah. Peningkatan kadar natrium dalam darah akan meningkatkan ekskresi kalsium ke dalam urine. Meningkatnya kadar kalsium urine memungkinkan pengendapan kalsium dalam urine, sehingga terbentuk batu ginjal. Batasi penggunaan suplemen kalsium dan vitamin D3 karena meningkatkan risiko pembentukan batu kalsium. Buang air kecil tepat waktu mempercepat pembilasan kuman yang masuk ke dalam saluran kemih dan substansi yang berpotensi mengendap membentuk batu ginjal.[4,8,20] Pencegahan pada Pasien yang Telah Mengalami Batu Ginjal Pada pasien yang sudah mengalami batu ginjal, tujuan pencegahan tidak hanya untuk mencegah terulangnya batu ginjal di kemudian hari, tetapi juga untuk mencegah tumbuhnya batu yang sudah ada. Pencegahan terutama terdiri dari modifikasi gaya hidup. Perbanyak konsumsi air putih dan buah dan sayur sambil mengurangi asupan natrium dianjurkan untuk semua penderita batu ginjal yang sudah terbentuk. Terapi medikamentosa ditambahkan jika pasien tidak berespon dengan perubahan diet atau tidak dapat mematuhi rekomendasi diet. Batu Kalsium Oksalat: Untuk mencegah kekambuhan batu kalsium oksalat, dokter perlu menurunkan konsentrasi faktor litogenik seperti kalsium dan oksalat, dan meningkatkan konsentrasi penghambat pembentukan batu seperti sitrat. Hal ini dapat dicapai dengan mengurangi asupan protein non dairy dan makanan tinggi oksalat dan suplemen vitamin C, serta fruktosa dan sukrosa. Terapi medikamentosa diberikan jika perubahan pola makan tidak mengurangi kekambuhan kolik.[23-26]`
  },
  {
    document_id: 'additional_document_4', 
    category: 'Muskuloskeletal',
    text: `Nyeri punggung bawah adalah sensasi nyeri yang dirasakan pada daerah tulang rusuk terbawah, pinggang, hingga bokong yang dapat menjalar hingga ke tungkai sampai batas lutut. Nyeri punggung bawah dapat terbagi menjadi akut dan kronik. Nyeri punggung bawah didefinisikan sebagai kronik bila terjadi selama lebih dari 3 bulan.[1,2] Patofisiologi nyeri punggung bawah berkaitan dengan mekanisme nyeri perifer dan melibatkan berbagai struktur anatomis. Etiologi nyeri punggung bawah dapat berupa gangguan mekanik, degeneratif, inflamasi, infeksi, hingga keganasan. Faktor risiko nyeri punggung bawah mencakup usia lebih tua, kebiasaan merokok, obesitas, riwayat nyeri punggung bawah sebelumnya, stres fisik, dan stres psikologi.[2-5] Young,Woman,Of,The,Low,Back,Pain Anamnesis dan pemeriksaan fisik memainkan peran penting dalam penegakan diagnosis nyeri punggung bawah. Anamnesis perlu menggali karakteristik nyeri, riwayat trauma, komorbiditas, dan pekerjaan serta kebiasaan. Pemeriksaan fisik pada pasien nyeri punggung bawah mencakup inspeksi, palpasi, range of motion, uji kekuatan, manuver provokatif, hingga pemeriksaan neurologis. Perhatikan tanda bahaya (red flags) seperti nyeri punggung yang disertai demam, nyeri tekan fokal, atau hilangnya sensasi perianal (saddle anesthesia), untuk dapat mendeteksi dini kemungkinan penyebab nyeri punggung yang serius, seperti abses epidural, osteomyelitis, atau sindrom cauda equina. Pemeriksaan penunjang yang digunakan dalam penegakan diagnosis adalah pencitraan radiologi dengan rontgen, computed tomography (CT), hingga magnetic resonance imaging (MRI). Penatalaksanaan bersifat multifaktorial, mengkombinasikan analgesik, olahraga, fisioterapi, dan cognitive behavioural therapy. Terapi medikamentosa mencakup obat antiinflamasi nonsteroid (OAINS) seperti naproxen, opioid seperti tramadol, antiepilepsi seperti pregabalin, antidepresan seperti duloxetine, pelemas otot seperti eperisone, atau kombinasi dari golongan obat tersebut. Meski begitu, belum ada satu obat spesifik yang menunjukkan performa yang baik untuk menghilangkan nyeri punggung bawah atau kekambuhannya, serta aman digunakan dalam durasi panjang.[1-4]
    Etiologi nyeri punggung bawah dapat terbagi menjadi lima kelompok, yaitu mekanik, degeneratif, inflamasi, infeksi, dan keganasan. Faktor risiko nyeri punggung bawah dapat berasal dari pekerjaan hingga gaya hidup.[2,4]

    Mekanik

    Sebagian besar penyebab mekanik nyeri punggung bawah adalah cedera pada verterbra, diskus intravertebrae, atau jaringan lunak. Spondylolisthesis dapat terjadi secara akut maupun kronik. Kondisi lain dapat berupa strain pada otot paraspinalis atau muskulus quadratus lumborum, herniasi diskus, dan kehamilan.[2,4]

    Degeneratif

    Penyakit degeneratif seperti osteoarthritis, termasuk pada sendi facet atau sacroiliac, stenosis spinalis, degenerative disc disease, dan fraktur kompresif terkait osteoporosis merupakan penyebab nyeri punggung bawah.[2,4]

    Inflamasi

    Inflamasi yang terkait nyeri punggung bawah terutama disebabkan oleh spondiloarthropati inflamasi, seperti ankylosing spondylitis. Selain itu, nyeri punggung bawah dapat disebabkan oleh sacroiliitis.[2,4]

    Infeksi

    Penyebab infeksi bisa mencakup infeksi pada vertebra atau diskus, maupun abses pada epidural atau jaringan lunak. Osteomyelitis bisa berasal dari penyebaran hematologi, paling umum adalah Staphylococcus dari area tindakan pembedahan atau penggunaan obat intravena. Penyebab infeksi lain mencakup tuberkulosis.[2,4]

    Keganasan

    Keganasan yang paling sering menyebabkan nyeri punggung bawah adalah metastasis kanker dari tempat lain, termasuk kanker paru, kanker payudara, kanker prostat, kanker ginjal, dan kanker kolon.

    Kondisi lain yang bisa menyebabkan nyeri punggung bawah adalah lesi litik pada vertebra atau kanker sumsum tulang. Komplikasi dari keganasan dapat berupa fraktur patologis ataupun kompresi saraf akibat ekstensi langsung dari pertumbuhan tumor.[2,4]

    Faktor Risiko

    Faktor risiko nyeri punggung bawah antara lain:

    Faktor risiko pekerjaan: pekerjaan fisik berat, pergerakan membungkuk (bending), gaya berputar (twisting), mengangkat (lifting)
    Faktor risiko psikososial: ansietas, depresi, stres, tingkat edukasi rendah, dan rasa ketidakpuasan dengan pekerjaan
    Faktor risiko gaya hidup individual: merokok, berat badan berlebih atau obesitas, konsumsi alkohol, kurang tidur, dan kurang aktivitas fisik
    Riwayat penyakit: cedera lalu lintas, penyakit pada tulang punggung[6,7]
    Tabel 1. Faktor Risiko Nyeri Punggung Bawah

    Kelompok	Faktor risiko
    Individu	Usia 44 - 75 tahun
    Riwayat nyeri punggung bawah sebelumnya
    Tinggi badan >170 cm
    Pubertas (dewasa muda >19 tahun)
    Kesehatan/kebiasaan	Merokok
    Obesitas
    Konsumsi alkohol
    Gangguan tidur
    Aktivitas fisik kurang
    Riwayat penyakit kronis
    Riwayat nyeri pada daerah lainnya
    Pekerjaan yang menyebabkan	Getaran seluruh tubuh
    Mengangkat beban >25 kg
    Sering mengangkat beban
    Duduk >2 jam
    Menyetir >2 jam
    Menarik >25 kg
    Berlutut >15 menit
    Jongkok >15 menit
    Posisi tangan di atas bahu hingga 15 menit
    Postur membungkuk >60o selama >5% waktu bekerja
    Berjalan/berdiri >2 jam
    Stres fisik	Bekerja dalam militer, kursi mobil kurang nyaman
    Stres psikologi	Kurangnya dukungan sosial dalam lingkungan kerja, kecemasan, kurang puas terhadap hidup, depresi, psikosomatik
    Evaluasi etiologi menjadi aspek paling utama dalam diagnosis nyeri punggung bawah. Dokter, terutama yang berada di layanan kesehatan primer, perlu membedakan antara nyeri punggung bawah nonspesifik dengan yang tidak. Dokter juga perlu menyingkirkan sumber patologi nonspinal dan memastikan nyeri berasal dari sistem muskuloskeletal.[1-4,9] Anamnesis Karakteristik nyeri merupakan poin penting dalam anamnesis nyeri punggung bawah untuk mengetahui kemungkinan etiologi. Nyeri akibat etiologi mekanik biasanya diperberat oleh posisi statis, seperti duduk atau berdiri, aktivitas long-lever atau penggunaan ekstremitas untuk mendorong atau menarik, dan posisi levered postures seperti membungkuk. Nyeri dapat berkurang dengan istirahat atau gerakan, seperti berjalan. Nyeri mekanikal dapat berkurang dengan istirahat atau berbaring, sedangkan pasien yang mengalami nyeri akibat vaskuler atau viseral biasanya kesulitan mencari posisi yang nyaman. Nyeri yang tidak berkurang dengan istirahat dapat menunjukkan etiologi yang lebih serius, seperti infeksi atau keganasan. Pada pasien dengan riwayat trauma, mekanisme cedera perlu digali lebih dalam. Riwayat nyeri punggung bawah sebelumnya, riwayat pekerjaan dan psikososial, serta keterbatasan dalam aktivitas juga perlu menjadi bagian dari anamnesis.[1-4,9] Red Flags Beberapa red flags yang perlu diwaspadai adalah riwayat keganasan, penurunan berat badan tanpa sebab jelas, kegagalan pengobatan setelah lebih dari 4–6 minggu, dan adanya paraparesis.[1-4,9] Nyeri Radikuler Karakteristik nyeri radikuler adalah adanya penyebaran dari punggung bawah dan bokong ke kaki sesuai dengan dermatoma. Penyebab tersering adalah herniasi diskus.[1-4,9] Sindroma Sendi Facet (Facet Joint Syndrome) Karakteristik nyeri pada sindrom ini adalah nyeri dengan atau tanpa nyeri alih ke bagian paha atau selangkangan, tanpa penyebaran radikuler, tidak di bagian pusat punggung dan nyeri di punggung lebih terasa dibandingkan dengan yang di kaki. Nyeri bertambah dengan hiperekstensi, rotasi, melekuk ke samping, dan berjalan menanjak.[1-4,9] Nyeri Sendi Sakroiliak Karakteristik nyeri sendi sakroiliak adalah gejala bertambah parah saat duduk atau saat berubah postur. Penyebab dapat berupa osteoarthritis sendi sakroiliak, ligamen terkilir (sprain) atau enthesopathy.[1-4,9] Stenosis Spina Lumbalis Karakteristik nyeri stenosis spina lumbalis adalah nyeri di garis median tubuh (midline), radikulopati dengan klaudikasi neurologis, kelemahan motorik, paraestesia, dan gangguan saraf sensoris. Eksaserbasi gejala terjadi jika pasien Berdiri lama, atau ekstensi lumbalis.[1-4,9] Nyeri Diskogenik Karakteristik nyeri ini adalah bersifat tidak spesifik, aksial, tanpa penyebaran radikuler, tanpa deformitas tulang belakang atau gangguan keseimbangan. Penyebab umumnya adalah degradasi di matriks nukleus pulposus atau fisura di annulus fibrosus.[1-4,9] Pemeriksaan Fisik Pemeriksaan fisik bagi nyeri punggung bawah meliputi inspeksi, palpasi, range of motion, uji kekuatan, manuver provokatif, dan pemeriksaan neurologis. Pemeriksaan neurologis sendiri terdiri dari sensorik, motorik, dan refleks. Perbedaan panjang tungkai, ketidakseimbangan pelvis, skoliosis, kepala dan bahu condong ke depan, dan kifosis dapat diperiksa melalui inspeksi.[1-4,9] Inspeksi Lakukan inspeksi punggung dan postur tubuh. Perhatikan adanya skoliosis atau hiperkifosis.[1-4,9] Palpasi dan Perkusi Nyeri saat palpasi atau perkusi dapat menandakan sebuah infeksi, fraktur kompresi, atau metastasis kanker.[1-4,9] Pemeriksaan Neurologi Pemeriksaan neurologis dilakukan untuk mengkonfirmasi adanya radikulopati, terutama di L5 dan S1. Dokter juga harus memeriksa ada tidaknya saddle anesthesia yang dapat mengarahkan pada diagnosis sindrom cauda equina. Indikator gangguan dari radiks saraf adalah: Nyeri kaki unilateral lebih terasa daripada nyeri punggung bawah Penyebaran nyeri ke kaki atau jari kaki Kebas dan parestesia di daerah atau penyebaran yang sama Tes Straight Leg Raising memperburuk rasa nyeri Gangguan neurologis yang terlokalisir di satu radiks saraf[1-4,9] Straight Leg Raise Pemeriksaan straight leg raise dilakukan dengan mengangkat tungkai pasien dalam posisi lurus hingga mencapai sudut 30 sampai 70 derajat, di mana nyeri pada kurang dari 60 derajat menandakan adanya herniasi diskus lumbar.[1-4,9] Waddell Signs Pemeriksaan Waddell signs dilakukan untuk mengetahui etiologi non-organik atau psikogenik. Pemeriksaan Waddell tidak akan menyebabkan nyeri dalam kondisi normal, misalnya rotasi panggul tanpa diikuti rotasi vertebra selain lumbal, tekanan ringan pada kepala, maupun perbandingan straight leg raise pada posisi duduk dan berbaring.[1-4,9] Pemeriksaan Lain Pemeriksaan kelenjar getah bening dapat dilakukan pada pasien yang dicurigai mengalami keganasan.[1-4,9] Diagnosis Banding Diagnosis banding yang perlu dipikirkan adalah adanya penyebab organik yang signifikan pada pasien dengan nyeri punggung bawah. Ini mungkin mencakup keganasan, fraktur, sindrom cauda equina, ataupun infeksi. Keganasan Keganasan tulang belakang perlu dicurigai pada pasien dengan riwayat keganasan, penurunan berat badan tanpa sebab jelas, kegagalan pengobatan setelah 4–6 minggu, serta pasien berusia di atas 50 tahun.[2-4] Fraktur Fraktur perlu dicurigai pada pasien dengan riwayat trauma atau osteoporosis. Kemungkinan adanya fraktur meningkat pada pasien dengan riwayat trauma signifikan, misalnya jatuh dari ketinggian atau kecelakaan lalu lintas. Nyeri dirasakan dalam onset tiba-tiba atau loading pain.[2-4] Infeksi Beberapa infeksi yang bisa menyebabkan keluhan nyeri punggung bawah adalah osteomyelitis vertebral, infeksi kanal spinalis, infeksi ruang diskus intervertebralis, maupun abses paraspinal. Kondisi ini dapat menyebabkan kecacatan pada pasien apabila tidak ditangani dengan cepat. Gejala yang perlu diwaspadai adalah adanya demam, riwayat terapi kortikosteroid atau imunosupresan, pengguna narkotik suntik, dan pasien HIV.[2-4] Sindroma Cauda Equina Sindroma cauda equina merupakan gangguan neuromuskular yang disebabkan oleh penyempitan pada kanal spinalis yang mengkompresi serabut saraf corda equina. Manifestasi klinis yang mengarah pada sindroma cauda equina adalah disfungsi kandung kemih dengan onset tiba-tiba, saddle anesthesia, kelemahan progresif tungkai bawah, defisit sensorik yang menyebar, gangguan gait, dan inkontinensia fekal.[2-4] Pemeriksaan Penunjang Kebanyakan pasien dengan nyeri punggung bawah mekanikal dan tidak memiliki red flags tidak memerlukan pemeriksaan penunjang. Pada pasien dengan kecurigaan penyebab yang lebih berat, pemeriksaan penunjang radiologis merupakan penunjang diagnosis awal. Selanjutnya, pemeriksaan penunjang dipilih berdasarkan arah diagnosis yang didapat dari anamnesis dan pemeriksaan fisik.[1-4,9] Pencitraan Radiologi Rontgen memiliki manfaat yang terbatas dalam evaluasi nyeri punggung bawah. Rontgen lateral dapat digunakan untuk mengevaluasi fraktur kompresi pada pasien lansia dan terkadang bisa menunjukkan adanya lesi litik. Meski begitu, kebanyakan patologi signifikan, termasuk trauma dengan gejala neurologi, akan memerlukan CT Scan dan MRI untuk menilai keadaan tulang, korda spinalis, dan saraf. Pemeriksaan dengan computed tomography (CT) lebih efektif dilakukan bila lokasi kelainan patologis atau neurologis pada vertebra telah diketahui. Mielografi dengan CT dapat digunakan pada kecurigaan patologi pada radiks saraf. Magnetic resonance imaging (MRI) merupakan modalitas yang dapat diandalkan bila lokasi kelainan pada vertebra belum diketahui, adanya kecurigaan kondisi patologis pada medula spinalis atau jaringan lunak, herniasi diskus, maupun infeksi atau keganasan. Pemeriksaan lain, seperti elektromiografi (EMG), tes somatosensory evoked potential (SSEP), atau blok selektif radiks saraf dapat dilakukan bila pencitraan dengan CT scan dan MRI tidak menunjukkan lokasi patologi yang jelas.[1-4,9] Pemeriksaan Laboratorium Tidak ada pemeriksaan laboratorium yang dapat spesifik menegakkan diagnosis etiologi nyeri punggung bawah. Pemeriksaan darah lengkap, C-Reactive Protein (CRP), laju endap darah (LED), hingga kultur darah dapat membantu penegakan diagnosis infeksi dan peradangan. Pemeriksaan faal hemostasis dan golongan darah mungkin diperlukan sebagai skrining pada pasien yang dicurigai membutuhkan tindakan operatif. Pemeriksaan tambahan lactate dehydrogenase (LDH) dapat bermanfaat untuk mengetahui kondisi yang menyebabkan turnover sumsum dengan cepat, misalnya leukemia.[1-4,9]
    Meski nyeri punggung bawah kemungkinan besar tidak mengancam nyawa, penatalaksanaan tetap diperlukan karena kondisi ini sangat mempengaruhi kualitas hidup. Saat ini, ada beragam jenis modalitas yang digunakan untuk penatalaksanaan nyeri punggung bawah, termasuk latihan peregangan, analgesik, dan pelemas otot.

    Dalam melakukan terapi untuk nyeri punggung bawah, dokter harus terlebih dahulu memastikan tidak adanya kondisi penyebab yang serius, seperti sindrom cauda equina, dengan melakukan skrining tanda bahaya pada pasien.[1-4,9]

    Medikamentosa

    Penggunaan obat-obatan untuk nyeri punggung bawah kronis disarankan hanya untuk jangka pendek, misalnya saat eksaserbasi akut. Hal ini karena sifat dari terapi medikamentosa adalah meredakan gejala, bukan penyebab dasar nyeri. Selain itu, penggunaan jangka panjang dapat menimbulkan banyak efek samping.[1-4,9]

    Obat Oral

    Penggunaan obat antiinflamasi nonsteroid (OAINS) selektif direkomendasikan dalam tata laksana nyeri punggung bawah sebagai terapi simptomatik. Obat yang dapat digunakan misalnya naproxen dan diklofenak.

    Obat lain yang juga sering digunakan adalah opioid, tetapi terdapat beberapa studi yang menunjukkan keterbatasan efikasinya. Selain itu, opioid juga berkaitan dengan berbagai efek samping, termasuk risiko penyalahgunaan.

    Obat lain yang sering digunakan adalah obat pelemas otot. Meski demikian, bukti yang mendukung efikasinya masih terbatas. Contoh obat pelemas otot yang bisa digunakan adalah eperisone.

    Obat antidepresan yang telah diteliti efektivitasnya adalah golongan trisiklik. Gabapentin meringankan gejala jangka pendek pada pasien dengan radikulopati. Selective serotonin reuptake inhibitors (SNRIs) dan obat antiepilepsi, seperti pregabalin, belum terbukti untuk membantu pasien nyeri punggung bawah kronis.[1-4,9]

    Obat Topikal

    Obat topikal adalah pilihan yang aman untuk meredakan gejala nyeri punggung bawah. Obat yang bisa digunakan contohnya capsaicin dan lidocaine topikal.[1-4,9]

    Tabel 1. Medikamentosa untuk Nyeri Punggung Bawah

    Kelas	Obat	Dosis
    Obat anti-inflamasi non steroid (OAINS)	Aspirin	325-650 mg setiap 4-6 jam
    Naproxen	250-550 mg setiap 12 jam
    Ibuprofen	400-800 mg setiap 6-8 jam
    Diclofenac	50 mg setiap 8 jam
    Indomethacin	25-50 mg setiap 8-12 jam
    Meloxicam	7,5-15 mg sekali sehari
    Celecoxib	200 mg sekali sehari
    Sediaan topikal	Capsaicin	Patch 0,025%, 0,03%, 0,0375%; maksimal 4 patch per hari
    Lidocaine	Patch 4% atau 5% per 12 jam ATAU salep 3-4 kali per hari
    Pelemas otot	Cyclobenzapine	5-10 mg setiap 8 jam
    Methocarbamol	750 mg setiap 6-8 jam
    Tizanidine	2-4 mg setiap 6-12 jam
    Opioid	Oxycodone	5 mg setiap 4 jam (pasien naif opioid)
    Oxycodone/paracetamol	10-15 mg setiap 3 jam (pasien toleran opioid)
    Hydrocodone/paracetamol	
    5 mg setiap 4 jam (pasien naif opioid)

    5-10 mg setiap 3 jam (pasien toleran opioid)

    Tramadol	50 mg setiap 6-8 jam
    Sumber: Michael Sintong Halomoan, Alomedika, 2023.[9]

    Terapi Injeksi

    Terapi berupa injeksi anestesi, kortikosteroid, atau obat lain dapat dilakukan langsung pada jaringan lunak, sendi facet, radiks saraf, atau epidural. Injeksi pada spinal bukan merupakan pilihan pertama, tetapi mungkin bermanfaat mengatasi nyeri akut atau eksaserbasi nyeri kronis. Injeksi plasebo juga dilaporkan bermanfaat.

    Beberapa contoh terapi injeksi spinal adalah blok intra-artikuler facet, blok medial branch, neurotomi radiofrekuensi medial branch, injeksi sendi sacroilliaca, injeksi epidural, adhesiolisis epidural, hingga vertebroplasti.[1-4,9]

    Pembedahan

    Terapi pembedahan dapat menjadi pilihan pada etiologi tertentu, seperti vertebra tidak stabil yang berbahaya secara neurologis akibat trauma, spondilolistesis, dan infeksi spinal. Pembedahan juga bisa bermanfaat pada kondisi adanya defisit neurologis progresif akibat kelainan struktur anatomis, seperti herniasi diskus, keganasan, fraktur, deformitas, atau stenosis berat.[1-4,9]

    Latihan

    Terdapat berbagai latihan yang dapat menjadi pilihan terapi suportif nyeri punggung bawah, termasuk latihan umum dan peningkatan ketahanan otot.

    Latihan Umum

    Latihan umum adalah latihan yang bertujuan mengembalikan atau meningkatkan kekuatan dan ketahanan kelompok otot pada batang tubuh, ekstremitas atas, dan ekstremitas bawah, termasuk latihan untuk mobilitas, fleksibilitas, dan aerobik.[2,3,10,11]

    Latihan Lainnya

    Latihan peningkatan kekuatan dan ketahanan otot batang tubuh bertujuan untuk mengembalikan atau meningkatkan kekuatan, ketahanan, atau tenaga kelompok otot batang tubuh.

    Latihan kontrol gerakan bertujuan untuk mengubah, mengembalikan, atau melatih kembali kontrol terhadap pergerakan fungsional, dengan umpan balik mengenai pola gerakan.

    Latihan aerobik bertujuan untuk mengembalikan atau meningkatkan kapasitas atau efisiensi sistem kardiovaskuler.

    Latihan multimodal menggabungkan dua atau lebih latihan yang dijelaskan sebelumnya.[2,3,10,11]

    Tabel 2. Pilihan Latihan Pada Berbagai Kondisi Nyeri Punggung Bawah

    Kondisi Nyeri Punggung Bawah	Latihan
    Nyeri punggung bawah akut	Latihan umum
    Aktivasi otot spesifik pada batang tubuh
    Nyeri punggung bawah akut dengan nyeri tungkai	Peningkatan kekuatan dan ketahanan otot batang tubuh
    Aktivasi otot spesifik pada batang tubuh
    Nyeri punggung bawah kronik	Latihan umum atau multimodal
    Peningkatan kekuatan dan ketahanan otot batang tubuh
    Aktivasi otot spesifik pada batang tubuh
    Aerobik
    Nyeri punggung bawah kronik dengan nyeri tungkai	Aktivasi otot spesifik pada batang tubuh
    Kontrol gerakan
    Nyeri punggung bawah kronik dengan gangguan kontrol gerakan	Aktivasi otot spesifik pada batang tubuh
    Kontrol gerakan
    Nyeri punggung bawah kronik pada lanjut usia >60 tahun	Latihan umum
    Nyeri punggung bawah post-operatif	Latihan umum
    Sumber: Michael Sintong Halomoan, Alomedika, 2023.[10]

    Fisioterapi

    Manual merupakan intervensi yang dilakukan langsung oleh fisioterapis dalam upaya mengurangi nyeri dan disabilitas akibat nyeri punggung bawah.[2,3,10]

    Mobilisasi Sendi

    Mobilisasi sendi merupakan gerakan terampil pasif yang dilakukan oleh fisioterapis dalam kecepatan dan amplitudo yang bervariasi pada daerah pergerakan sendi. Teknik dapat berupa thrust dan non-thrust.[2,3,10]

    Mobilisasi Jaringan Lunak

    Mobilisasi jaringan lunak merupakan gerakan terampil pasif yang dilakukan oleh fisioterapis pada jaringan lunak, termasuk fascia, otot, dan ligamen. Teknik termasuk myofascial release, terapi trigger point, atau strain/counterstrain.[2,3,10]

    Mobilisasi Jaringan Saraf

    Mobilisasi jaringan saraf merupakan Gerakan terampil yang dilakukan oleh fisioterapis untuk meningkatkan keseimbangan pergerakan jaringan sekitar saraf.[2,3,10]

    Manual Lainnya

    Pijat adalah teknik menggunakan tangan yang meningkatkan relaksasi pada otot. Selain itu, bisa digunakan dry needle, yakni intervensi menggunakan jarum filiform tipis untuk menembus kulit dan merangsang trigger point miofasial serta jaringan otot dan ikat sebagai manajemen nyeri dan gangguan pergerakan.[2,3,10]

    Tabel 3. Pilihan Manual Pada Berbagai Kondisi Nyeri Punggung Bawah

    Kondisi nyeri punggung bawah	Manual
    Nyeri punggung bawah akut	Mobilisasi sendi
    Pijat
    Mobilisasi jaringan lunak
    Nyeri punggung bawah kronik	Mobilisasi sendi
    Mobilisasi jaringan lunak
    Pijat
    Dry needle
    Mobilisasi jaringan saraf
    Prognosis nyeri punggung bawah dipengaruhi oleh intensitas dan penjalaran nyeri. Komplikasi nyeri punggung bawah yang utama adalah penurunan kualitas hidup akibat keterbatasan pada aktivitas harian.[2-4,9] Komplikasi Nyeri punggung bawah dapat menimbulkan nyeri kronik yang menurunkan kualitas hidup pasien atau disabilitas. Selain itu, tergantung dari penyebabnya, nyeri punggung bawah juga bisa menimbulkan deformitas, defisit motorik dan sensorik, serta gangguan pencernaan atau kemih. Komplikasi sosial dapat berupa penurunan aktivitas dan pendapatan. Bila penyebab nyeri adalah fraktur di sendi sakroiliak, fraktur dapat berlanjut ke fraktur komplit di panggul bila tidak dilakukan penanganan. Herniasi diskus dapat menghasilkan komplikasi sindroma cauda equina yang dapat menyebabkan disabilitas permanen. Terdapat juga beberapa komplikasi akibat tata laksana nyeri punggung bawah kronis. Penggunaan obat-obatan seperti opioid dapat menyebabkan penyalahgunaan. Prosedur intervensi, seperti injeksi anastesi atau kortikosteroid, dapat menyebabkan kelemahan ekstremitas bawah, insomnia, nyeri kepala, dan gangguan elektrolit. Risiko komplikasi pembedahan mencakup cedera saraf, robekan pada dura mater, infeksi, atau degenerasi diskus.[2-4,9] Prognosis Prognosis nyeri punggung bawah dipengaruhi oleh berbagai faktor. Riwayat nyeri punggung bawah sebelumnya, intensitas nyeri lebih tinggi, dan adanya penjalaran nyeri ke tungkai dikaitkan dengan prognosis yang lebih buruk. Prognosis juga dipengaruhi oleh aktivitas fisik dan kebiasaan, seperti merokok. Risiko disabilitas meningkat pada pasien nyeri punggung bawah yang disertai depresi atau kelainan psikologis lain. Faktor sosial lain, seperti tingkat edukasi rendah, pekerjaan yang membutuhkan gerakan fisik, pendapatan yang tidak sesuai beban kerja, dan ketidakpuasan terhadap pekerjaan memperburuk prognosis. Namun, sebagian besar kasus nyeri punggung bawah dapat membaik tanpa menyebabkan gangguan.[2-4,9]
    Edukasi dan promosi kesehatan pada nyeri punggung bawah harus mencakup cara menghindari eksaserbasi, cara mengangkat barang yang aman di tempat kerja dan di rumah, dan postur tubuh yang baik untuk menghindari tekanan kepada tulang belakang.[2-4,9-11] Edukasi Pasien Pada pasien nyeri punggung bawah, lakukan edukasi untuk mengurangi berat badan bila pasien memiliki berat badan berlebih atau obesitas. Anjurkan untuk menggunakan sepatu rata (bukan hak tinggi) dengan sol sepatu yang nyaman dan empuk. Hindari pergerakan yang mendadak atau tekanan berlebih, serta kurangi stres atau tekanan pikiran psikologis.[2-4,9-11] Postur Postur yang baik untuk mencegah eksaserbasi sangat penting. Anjurkan pasien berdiri tegak dengan kepala menghadap ke depan dan punggung lurus. Seimbangkan berat badan antara kedua kaki dan saat berdiri pertahankan kaki lurus. Saat duduk, anjurkan duduk tegak dengan beban tekanan pada bagian tengah punggung bawah, lutut dan pinggang sederajat dan telapak kaki di lantai. Penggunaan bantal kecil atau handuk yang tergulung dapat membantu menguatkan bagian tengah punggung bawah.[2-4,9-11] Gaya Hidup Anjurkan pasien agar tidur menggunakan kasur yang cukup kuat menyangga berat badan, terutama di bahu dan bokong agar tulang belakang pada posisi anatomis Olahraga seperti berjalan atau berenang dapat menguatkan otot. Beberapa gerakan olahraga sehari-hari di rumah dapat meringankan gejala nyeri dan menguatkan otot, di antaranya pelvic tilt, hip flexors, tail wag, dan lumbar rotation. Aktivitas seperti yoga atau pilates dengan pelatih yang berpengalaman dapat meningkatkan fleksibilitas dan kekuatan punggung bawah.[2-4,9-11] Mengangkat Barang Saat mengangkat barang, mulai dengan posisi yang baik. Kedua kaki terpisah dengan satu kaki diposisikan agak ke depan untuk mempertahankan keseimbangan. Saat mengangkat, beban barang sebaiknya ditahan oleh kedua kaki. Bengkokkan lutut, punggung, dan panggul, tetapi jangan jongkok atau membungkukkan badan. Kuatkan otot perut untuk mendorong bagian pelvis ke dalam saat mengangkat, jangan luruskan kaki sebelum benda sudah terangkat dengan baik. Saat membawa barang, pertahankan beban di dekat pinggang dengan bagian terberat di dekat tubuh. Jangan menggerakkan badan ke samping atau miring, pertahankan wajah dan panggul tetap searah. Jika ada benda berat, lebih baik mendorong daripada menarik. Bagi beban yang dibawa sama rata antara kanan dan kiri.[2-4,9-11] Nyeri Punggung Bawah Kronik Pada pasien dengan nyeri punggung bawah kronik, jelaskan bahwa terapi multimodal diperlukan. Ini mencakup kombinasi olahraga, terapi medikamentosa, dan cognitive behavioural therapy atau acceptance and commitment therapy.[2-4,9-11] Upaya Pencegahan dan Pengendalian Penyakit Seperti penyakit tidak menular lainnya, pencegahan nyeri punggung bawah perlu dilakukan dengan perubahan gaya hidup. Upaya pencegahan nyeri punggung bawah dapat dilakukan melalui program penyakit tidak menular Kementerian Kesehatan Republik Indonesia berupa CERDIK yang merupakan akronim dari: Cek kesehatan secara berkala Enyahkan asap rokok Rajin aktivitas fisik Diet seimbang Istirahat cukup Kelola stres Adapun upaya pengendalian nyeri punggung bawah dapat dilakukan dengan program penyakit tidak menular Kementerian Kesehatan Republik Indonesia berupa PATUH yang merupakan akronim dari: Periksa kesehatan secara rutin dan ikuti anjuran dokter Atasi penyakit dengan pengobatan yang tepat dan teratur Tetap diet dengan gizi yang seimbang Upayakan aktivitas fisik dengan aman Hindari asap rokok, alkohol, dan zat karsinogenik[12,13]`
  }
];

// Helper function to retry operation on ECONNRESET
async function retryOperation<T>(operation: () => Promise<T>, maxRetries: number = 3, delayMs: number = 2000): Promise<T> {
  let attempt = 0;
  while (attempt < maxRetries) {
    try {
      return await operation();
    } catch (error: any) {
      if (error.message && error.message.includes('ECONNRESET')) {
        attempt++;
        console.warn(`ECONNRESET encountered. Retrying attempt ${attempt}/${maxRetries} in ${delayMs}ms...`);
        await new Promise(res => setTimeout(res, delayMs));
      } else {
        throw error;
      }
    }
  }
  throw new Error(`Operation failed after ${maxRetries} retries due to recurring ECONNRESET errors.`);
}

async function insertTextData() {
  const collection = await astraDb.collection(COLLECTION_NAME);

  for (const doc of dataToInsert) {
    const chunks = await splitter.splitText(doc.text);
    let i = 0;
    for (const chunk of chunks) {
      // Generate embedding for this chunk
      const { data } = await openai.embeddings.create({ input: chunk, model: 'text-embedding-3-small' });

      // Construct the record with your custom fields
      const record = {
        id: uuidv4(),
        document_id: doc.document_id,
        chunk_id: `${doc.document_id}-${i}`,
        text: chunk,
        category: doc.category,
        $vector: data[0]?.embedding
      };

      // Insert the record with retry
      await retryOperation(() => collection.insertOne(record));
      i++;
    }
  }

  console.log('All custom text data inserted successfully.');
}

insertTextData().catch(console.error);
