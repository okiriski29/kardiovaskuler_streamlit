#Import Library
import streamlit as st
import numpy as np
import pandas as pd
import datetime
import time
import pickle
from datetime import date, timedelta
from streamlit_option_menu import option_menu
from modeling import final_accuracy
#Dapatkan waktu sekarang
current_time = datetime.datetime.now()
st.set_page_config(layout="wide")
# Membuka dan memuat model
with open('modelkardio.pkl', 'rb') as file:
    model = pickle.load(file)
#Membuat Sidebar
with st.sidebar:
    st.image("logo.png")
    selected = option_menu("Main Menu", ["🏠 Beranda",'📖 Informasi', '🩺 Prediksi', '💾 Dataset', '📊 Visualisasi'], 
        icons=['none', 'none', 'none', 'none', 'none'], menu_icon="cast", default_index=0)
#Membuat Halaman Home
if selected=='🏠 Beranda':
    # --- HALAMAN BERANDA ---
    # Teks header utama sesuai gambar Anda
    st.markdown("# Selamat Datang Di Website Cardiovascular Care 🫀")
    st.write("---")

    # 1. Ringkasan Pengantar (Hero Section)
    st.markdown("""
    Sistem berbasis Pembelajaran Mesin (*Machine Learning*) ini dirancang untuk membantu Anda 
    melakukan deteksi dini dan penilaian mandiri terhadap risiko penyakit kardiovaskular (jantung dan pembuluh darah). 
    Sistem bekerja dengan menganalisis kombinasi data kesehatan, seperti tekanan darah, kadar kolesterol, gula darah, 
    riwayat penyakit keluarga, serta pola gaya hidup sehari-hari, termasuk aktivitas fisik dan kebiasaan merokok. 
    Hasil evaluasi yang diberikan dapat membantu pengguna memahami tingkat risiko kardiovaskular sehingga dapat melakukan langkah pencegahan dan menjaga kesehatan jantung lebih awal.
    """)
    st.markdown("""
    <style>
    .metric-container{
        display:flex;
        gap:20px;
        margin-top:20px;
        margin-bottom:25px;
        flex-wrap:wrap;
    }

    .metric-card{
        flex:1 1 300px;
        background: linear-gradient(135deg,#0F4C81,#1F78D1);
        padding:35px 25px;
        border-radius:25px;
        color:white;
        text-align:center;
        position:relative;
        overflow:hidden;
        transition:0.3s;
        box-shadow:0px 8px 20px rgba(0,0,0,0.15);
    }

    .metric-card:hover{
        transform:translateY(-8px);
        box-shadow:0px 12px 25px rgba(0,0,0,0.25);
    }

    .metric-card::before{
        content:'';
        position:absolute;
        width:180px;
        height:180px;
        background:rgba(255,255,255,0.08);
        border-radius:50%;
        top:-60px;
        right:-60px;
    }

    .metric-icon{
        font-size:50px;
        margin-bottom:10px;
    }

    .metric-value{
        font-size:42px;
        font-weight:bold;
        margin-bottom:10px;
    }

    .metric-title{
        font-size:20px;
        font-weight:600;
        margin-bottom:8px;
    }

    .metric-desc{
        font-size:15px;
        opacity:0.9;
        line-height:1.6;
    }

    </style>
    <div class="metric-container">

    <div class="metric-card">
        <div class="metric-icon">🌍</div>
        <div class="metric-value">#1</div>
        <div class="metric-title">Penyebab Kematian Global dan Nasional</div>
        <div class="metric-desc">
            Penyakit Kardiovaskular Menjadi Penyebab Kematian Tertinggi Di Dunia dan Di Indonesia.
        </div>
    </div>

    <div class="metric-card">
        <div class="metric-icon">🩺</div>
        <div class="metric-value">0</div>
        <div class="metric-title">Deteksi Dini</div>
        <div class="metric-desc">
            Tidak Adanya Fasilitas Deteksi Dini Penyakit Kardiovaskular Yang Diberikan Kepada Masyarakat
        </div>
    </div>

    <div class="metric-card">
        <div class="metric-icon">🚨</div>
        <div class="metric-value">19,8 Juta</div>
        <div class="metric-title">Kematian</div>
        <div class="metric-desc">Rantai Kematian Akibat Penyakit Kardiovaskular Harus Diputus</div>
    </div>

    </div>
    """, unsafe_allow_html=True)
    st.write("---")

    # 3. Panduan Alur Penggunaan Aplikasi
    st.markdown("### 🧭 Cara Melakukan Cek Risiko Kardiovaskular:")

    col_step1, col_step2, col_step3 = st.columns(3)

    with col_step1:
        st.info("""
        **1. Pilih Menu Prediksi**
        Buka panel menu di samping kiri layar (*sidebar*), lalu klik menu **🩺 Prediksi**.
        """)

    with col_step2:
        st.info("""
        **2. Isi Data Kesehatan**
        Masukkan parameter tubuh Anda secara akurat (seperti data tekanan darah, kolesterol, dan gaya hidup).
        """)

    with col_step3:
        st.info("""
        **3. Lihat Hasil Evaluasi**
        Tekan tombol prediksi untuk melihat hasil analisis risiko beserta rangkuman data riwayat Anda.
        """)

    st.write("---")

    # 4. Edukasi Singkat Mengenai Faktor Risiko
    with st.expander("💡 Pelajari Faktor Risiko Utama Penyakit Jantung dan Pembuluh Darah(Kardiovaskular)"):
        st.markdown("""
        Penyakit kardiovaskular sering kali berkembang tanpa gejala awal yang disadari. Berikut adalah parameter kritis yang perlu Anda pantau:
        * **Tekanan Darah Tinggi (Hipertensi):** Beban kerja berlebih pada pembuluh darah memperbesar risiko kerusakan arteri.
        * **Kadar Kolesterol Tinggi:** Dapat memicu penumpukan plak (aterosklerosis) yang menyumbat aliran darah ke jantung.
        * **Diabetes:** Kadar gula darah tinggi dapat merusak pembuluh darah dan meningkatkan risiko penyakit jantung.
        * **Riwayat Keluarga:** Faktor genetik dari keluarga dengan riwayat penyakit jantung dapat meningkatkan risiko kardiovaskular.
        * **Obesitas:** Berat badan berlebih membuat kerja jantung lebih berat dan meningkatkan risiko hipertensi serta kolesterol tinggi.
        * **Gaya Hidup:** Merokok, kurang aktivitas fisik (olahraga)
        """)

    # 5. Catatan / Disclaimer Medis Khas Aplikasi Kesehatan
    st.warning("""
    ⚠️ **Catatan Penting:** Hasil dari aplikasi ini bersifat sebagai skrining awal/edukasi 
    dan tidak menggantikan diagnosis medis formal dari dokter spesialis Jantung dan Pembuluh Darah. Jika Anda merasakan gejala nyeri dada atau sesak napas, 
    segera hubungi layanan medis darurat atau fasilitas kesehatan terdekat.
    """)

    st.balloons()
if selected=='📖 Informasi':
    st.markdown("""
    <style>
    .main-card{
        background: linear-gradient(135deg,#ffffff,#f5f9ff);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    .title{
        text-align:center;
        font-size:38px;
        font-weight:bold;
        color:#0B3C5D;
        margin-bottom:10px;
    }

    .subtitle{
        text-align:center;
        font-size:18px;
        color:#555;
        margin-bottom:30px;
    }

    .info-box{
        background:#FFE7D1;
        padding:15px;
        border-radius:12px;
        margin-bottom:12px;
        font-size:17px;
        color:#333;
        border-left:6px solid #FF914D;
    }

    .section-title{
        font-size:28px;
        font-weight:bold;
        color:#0B3C5D;
        margin-top:20px;
        margin-bottom:15px;
    }

    .paragraph{
        text-align:justify;
        font-size:17px;
        line-height:1.9;
        color:#333;
    }

    .warning{
        background:linear-gradient(135deg,#ffefef,#ffe3e3);
        padding:20px;
        border-radius:15px;
        border-left:7px solid red;
        margin-top:20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-card">

    <div class="title">
    🫀 Informasi Terkait Penyakit Kardiovaskular
    </div>

    <div class="subtitle">
    Kenali faktor risiko dan pentingnya menjaga kesehatan jantung sejak dini
    </div>

    <div class="section-title">
    📖 Apa Itu Kardiovaskular?
    </div>

    <div class="paragraph">
    Kardiovaskular adalah istilah yang merujuk pada sistem jantung dan pembuluh darah, serta penyakit yang berkaitan dengannya. Kardiovaskular merupakan sebuah kondisi di mana terjadi penyempitan atau penyumbatan pembuluh darah yang dapat menyebabkan serangan jantung, nyeri dada (angina), atau stroke. Penyakit kardiovaskuler termasuk kondisi kritis yang butuh penanganan segera. Pasalnya, jantung adalah organ vital yang berfungsi untuk memompa darah ke seluruh tubuh. Jika jantung bermasalah, peredaran darah dalam tubuh bisa terganggu. Tanpa pertolongan medis yang sesuai, penyakit kardiovaskuler bisa mengancam jiwa dan menyebabkan kematian.
    </div>

    <br>

    <div class="section-title">
    🔍 Organ Utama Sistem Kardiovaskular
    </div>

    <div class="info-box">
    ❤️ <b>Jantung</b> → Memompa darah ke seluruh tubuh
    </div>

    <div class="info-box">
    🩸 <b>Arteri</b> → Membawa darah dari jantung
    </div>

    <div class="info-box">
    🔄 <b>Vena</b> → Mengalirkan darah kembali ke jantung
    </div>

    <div class="info-box">
    🌐 <b>Kapiler</b> → Pembuluh darah kecil untuk distribusi oksigen dan nutrisi
    </div>
    <div class="section-title">
    🧬 Jenis-Jenis Penyakit Kardiovaskular
    </div>

    <div class="info-box">
    ❤️‍🩹 <b>Jantung Koroner</b> →  Penyakit jantung koroner terjadi ketika aliran darah kaya oksigen ke otot jantung tersumbat atau berkurang
    </div>

    <div class="info-box">
    🧠 <b>Stroke</b> → - Stroke adalah kondisi saat suplai darah ke bagian otak terputus, yang dapat menyebabkan kerusakan otak dan kemungkinan kematian
    </div>

    <div class="info-box">
    💗 <b>Aritmia</b> → Kondisi ini terjadi ketika detak jantung berlangsung dengan tidak teratur. Detak jantung bisa terjadi dengan sangat cepat atau sangat lambat
    </div>

    <div class="info-box">
    💔 <b>Serangan Jantung</b> → Serangan jantung bisa terjadi akibat terputusnya aliran darah menuju otot jantung secara tiba-tiba
    </div>

    <div class="info-box">
    🫀 <b>Gagal Jantung</b> → Kondisi ini terjadi ketika jantung tidak mampu memompa darah untuk memenuhi kebutuhan tubuh
    </div>

    </div>
    """, unsafe_allow_html=True)
    
    # ================= CSS =================
    st.markdown("""
    <style>

    .main {
        background-color: #f5f7fb;
    }

    .hero{
        background: linear-gradient(135deg,#0F4C81,#1F78D1);
        padding: 45px;
        border-radius: 25px;
        color: white;
        text-align:center;
        margin-bottom:30px;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.15);
    }

    .hero h1{
        font-size:48px;
        margin-bottom:10px;
    }

    .hero p{
        font-size:19px;
        line-height:1.8;
    }

    .card{
        background:white;
        padding:25px;
        border-radius:20px;
        box-shadow:0px 4px 12px rgba(0,0,0,0.08);
        margin-bottom:25px;
    }

    .card-title{
        font-size:30px;
        font-weight:bold;
        color:#0F4C81;
        margin-bottom:15px;
    }

    .text{
        font-size:17px;
        text-align:justify;
        line-height:1.9;
        color:#333;
    }

    .info-box{
        background:#FFF3E8;
        padding:18px;
        border-radius:15px;
        margin-bottom:15px;
        border-left:6px solid #FF914D;
        font-size:17px;
    }

    .risk-card{
        background:linear-gradient(135deg,#FFECEC,#FFF5F5);
        padding:20px;
        border-radius:18px;
        margin-bottom:15px;
        border-left:7px solid #E53935;
    }

    .prevention{
        background:linear-gradient(135deg,#E8FFF1,#F4FFF8);
        padding:20px;
        border-radius:18px;
        border-left:7px solid #2EAF62;
        margin-top:10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ================= METRIC =================
    
    # ================= FAKTOR RISIKO =================
    st.markdown("""
    <div class="card">

    <div class="card-title">
    ⚠️ Faktor Risiko Penyakit Kardiovaskular
    </div>

    <div class="risk-card">
    <b>Tekanan Darah Tinggi (Hipertensi)</b><br>
    Tekanan darah tinggi membuat jantung bekerja lebih keras dan meningkatkan risiko kerusakan pembuluh darah.
    </div>

    <div class="risk-card">
    <b>Kolesterol Tinggi</b><br>
    Kolesterol berlebih dapat menyebabkan penumpukan plak pada pembuluh darah.
    </div>

    <div class="risk-card">
    <b>Diabetes</b><br>
    Kadar gula darah tinggi dapat merusak pembuluh darah dan meningkatkan risiko penyakit jantung.
    </div>

    <div class="risk-card">
    <b>Obesitas</b><br>
    Berat badan berlebih meningkatkan risiko hipertensi, diabetes, dan kolesterol tinggi.
    </div>

    <div class="risk-card">
    <b>Merokok & Kurang Aktivitas Fisik</b><br>
    Kebiasaan merokok dan kurang olahraga dapat memperburuk kesehatan jantung.
    </div>

    <div class="risk-card">
    <b>Riwayat Keluarga</b><br>
    Faktor genetik dapat meningkatkan kemungkinan terkena penyakit kardiovaskular.
    </div>

    </div>
    """, unsafe_allow_html=True)

    # ================= GEJALA =================
    st.markdown("""
    <div class="card">

    <div class="card-title">
    🚨 Gejala Umum Penyakit Kardiovaskular
    </div>

    <div class="text">
    <ul style="line-height:2;">
    <li>Nyeri dada atau rasa tertekan di dada</li>
    <li>Sesak napas</li>
    <li>Detak jantung tidak teratur</li>
    <li>Mudah lelah</li>
    <li>Pusing atau kehilangan kesadaran</li>
    <li>Pembengkakan pada kaki</li>
    </ul>
    </div>

    </div>
    """, unsafe_allow_html=True)

    # ================= PENCEGAHAN =================
    st.markdown("""
    <div class="card">

    <div class="card-title">
    ✅ Cara Pencegahan
    </div>

    <div class="prevention">
    <ul style="line-height:2;">
    <li>Rutin berolahraga minimal 30 menit setiap hari</li>
    <li>Mengurangi makanan tinggi garam dan lemak</li>
    <li>Memperbanyak konsumsi buah dan sayur</li>
    <li>Berhenti merokok dan menghindari alkohol</li>
    <li>Menjaga berat badan ideal</li>
    <li>Melakukan pemeriksaan kesehatan secara berkala</li>
    </ul>
    </div>

    </div>
    """, unsafe_allow_html=True)
if selected=='💾 Dataset':
    st.subheader("Dataset Kardiovaskular")
    dataset = pd.read_csv('Kardio.csv')
    st.dataframe(dataset)
    st.download_button("Download Dataset", data='Kardio.csv', file_name="Kardio.csv", type='primary')
    st.write(f"Akurasi model dataset ini sebesar **{final_accuracy*100:.2f}**%")

if selected=='📊 Visualisasi':
    st.title(':chart_with_upwards_trend: Visualisasi Data ')
    st.header("1. Heatmap Correlation")
    st.image("image/heatmap correlation.png", caption="Heatmap Correlation Features")
    st.header("2. Distribusi Target")
    st.image("image/distribusi kardio.png", caption="Heatmap Correlation Features")
    st.header("3. Distribusi Usia")
    st.image("image/distribusi usia.png", caption="Heatmap Correlation Features")
    st.header("4. Feature Importances")
    st.image("image/feature importances.png", caption="Heatmap Correlation Features")
    st.header("5. Confusion Matrix")
    st.image("image/confusion matrix.png", caption="Confusion Matrix")
    st.header("6. Distribusi Jenis Kelamin")
    st.image("image/distribusi jk.png", caption="Distribusi Jenis Kelamin")
    st.header("7. Distribusi Sistolik Pasien Kardio")
    st.image("image/distribusi sistolik kardio.png", caption="Distribusi Sistolik Pasien Kardio")
    st.header("8. Distribusi Sistolik Pasien Non Kardio")
    st.image("image/distribusi sistolik no kardio.png", caption="Distribusi Sistolik Pasien Non Kardio")
    
#Membuat Halaman Prediksi
if selected=='🩺 Prediksi':
    st.header(" :clipboard: Cek Risiko Kamu Terkena Penyakit Kardiovaskular")
    col1, col2, col3 = st.columns([2,1,1])
    col4, col5, col6, col7 = st.columns(4)
    jk = ("Laki-laki", "Perempuan")
    option = ("Tidak", "Iya")
    options = list(range(len(jk)))
    with col1:
        nama = st.text_input("Nama Anda:", placeholder="Masukkan Nama Anda" )
    with col2:
        start_date = date.today() - timedelta(days=100*365)  # 50 years ago from today
        end_date = date.today() 
        born = st.date_input("Tanggal Lahir",min_value=start_date, max_value=end_date)
        def calculate_age(born):
            today = date.today()
            return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    with col3:
        umur = st.text_input("Umur Anda", calculate_age(born), disabled=True)
        umur = int(umur)
    with col4:
        jk = st.selectbox("Jenis Kelamin", options, format_func=lambda x: jk[x])
        tinggi_badan = st.number_input("Tinggi Badan",value=None, min_value=125,max_value=565,step=1, placeholder="(cm)")
        berat_badan=st.number_input("Berat Badan",value=None, min_value=40,max_value=150,step=1, placeholder="(kg)")
        st.write(' ')
    with col5:
        sistolik = st.number_input("Tekanan Sistolik",value=None, min_value=70,max_value=250,step=10, placeholder="(mmHg)")
        diastolik = st.number_input("Tekanan Diastolik",value=None, min_value=40,max_value=160,step=10, placeholder="(mmHg)")
        kolesterol = st.selectbox("Kolesterol", options, format_func=lambda x: option[x])
    with col6:
        diabetes = st.selectbox("Diabetes", options, format_func=lambda x: option[x])
        riwayat = st.selectbox("Riwayat Keluarga", options, format_func=lambda x: option[x])
    with col7:
        merokok = st.selectbox("Merokok", options, format_func=lambda x: option[x])
        olahraga = st.selectbox("Olahraga", options, format_func=lambda x: option[x])
    new_data_input = {
        'umur': [umur],
        'jk': [jk],
        'tinggi': [tinggi_badan],
        'berat': [berat_badan],
        'sistolik': [sistolik],
        'diastolik': [diastolik],
        'kolesterol': [kolesterol],
        'diabetes': [diabetes],
        'riwayat': [riwayat],
        'merokok': [merokok],
        'olahraga': [olahraga]
    }
    # Data asli untuk prediksi
    new_data_df = pd.DataFrame(new_data_input)
    prediksi = model.predict(new_data_df)
    prediksi_proba = model.predict_proba(new_data_df)
    
    print("\nPrediksi untuk data baru:")

    if prediksi[0] == 1:
        print("Pasien ini diprediksi memiliki risiko TINGGI terkena penyakit kardiovaskular")
    else:
        print("Pasien ini diprediksi memiliki risiko RENDAH penyakit kardiovaskular")

    print("\nProbabilitas:")
    print(f"Tidak Kardio : {prediksi_proba[0][0]*100:.2f}%")
    print(f"Kardio       : {prediksi_proba[0][1]*100:.2f}%")


    status = ''
    if st.button("Prediksi", type="primary"):
        if(jk==1):
            jk = "Laki-laki"
        else:
            jk = "Perempuan"
        if(kolesterol==1):
            kolesterol = "Iya"
        else:
            kolesterol = "Tidak"
        if(diabetes==1):
            diabetes = "Iya"
        else:
            diabetes = "Tidak"
        if(riwayat==1):
            riwayat = "Iya"
        else:
            riwayat = "Tidak"
        if(merokok==1):
            merokok = "Iya"
        else:
            merokok = "Tidak"
        if(olahraga==1):
            olahraga = "Iya"
        else:
            olahraga = "Tidak"
        if(nama==''):
            st.warning('Data belum diinput', icon="⚠️")
        elif (prediksi)==1:
            status = "Risiko Tinggi"
            with st.spinner('Sedang Memprediksi...'):
                time.sleep(3)
            st.subheader("🩺 Hasil Prediksi :")
            st.error("Kamu berisiko TINGGI terkena penyakit kardiovaskular", icon='🚨')
            st.write("### 📈 **Probabilitas Risiko**")
            # HTML dan CSS Kustom untuk kartu probabilitas
            st.markdown(f"""
            <div style="display: flex; gap: 15px; margin: 15px 0px;">
                <div style="background-color: #FFEBEE; padding: 20px; border-radius: 10px; flex: 1; border-left: 5px solid #C62828; ">
                    <p style="margin: 0; color: #C62828; font-weight: bold; font-size: 14px;">RISIKO TINGGI</p>
                    <p style="margin: 5px 0 0 0; font-size: 28px; font-weight: bold; color: #B71C1C; ">{prediksi_proba[0][1]*100:.2f}%</p>
                </div>
                <div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px; flex: 1; border-left: 5px solid #2E7D32;">
                    <p style="margin: 0; color: #2E7D32; font-weight: bold; font-size: 14px;">RISIKO RENDAH</p>
                    <p style="margin: 5px 0 0 0; font-size: 28px; font-weight: bold; color: #1B5E20;">{prediksi_proba[0][0]*100:.2f}%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.write("### 📋 **Tabel Rangkuman Data Pasien**")
            df = pd.DataFrame(
            [
                {"waktu":current_time, "nama":nama, "umur":umur, "tinggi_badan":tinggi_badan, "berat_badan":berat_badan , "jenis_kelamin":jk, "sistolik":sistolik, "diastolik":diastolik,  "kolesterol": kolesterol, "diabetes": diabetes, "riwayat":riwayat, "merokok": merokok, "olahraga":olahraga, "hasil": status}
            ]
            )
            # Styling warna berdasarkan hasil
            def highlight_hasil(val):
                if val == "Risiko Tinggi":
                    return "background-color: #ff4b4b; color: white;"
                elif val == "Risiko Rendah":
                    return "background-color: #28a745; color: white;"
                return ""

            styled_df = df.style.map(
                highlight_hasil,
                subset=["hasil"]
            )

            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
        elif (prediksi)==0:
            status = "Risiko Rendah"
            with st.spinner('Sedang Memprediksi...'):
                time.sleep(3)
            st.subheader("🩺 Hasil Prediksi :")
            st.success("Kamu berisiko RENDAH terkena penyakit kardiovaskular", icon='💚')
            st.write("### 📈 **Probabilitas Risiko**")
            # HTML dan CSS Kustom untuk kartu probabilitas
            st.markdown(f"""
            <div style="display: flex; gap: 15px; margin: 15px 0px;">
                <div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px; flex: 1; border-left: 5px solid #2E7D32;">
                    <p style="margin: 0; color: #2E7D32; font-weight: bold; font-size: 14px;">RISIKO RENDAH</p>
                    <p style="margin: 5px 0 0 0; font-size: 28px; font-weight: bold; color: #1B5E20;">{prediksi_proba[0][0]*100:.2f}%</p>
                </div>
                <div style="background-color: #FFEBEE; padding: 20px; border-radius: 10px; flex: 1; border-left: 5px solid #C62828;">
                    <p style="margin: 0; color: #C62828; font-weight: bold; font-size: 14px;">RISIKO TINGGI</p>
                    <p style="margin: 5px 0 0 0; font-size: 28px; font-weight: bold; color: #B71C1C;">{prediksi_proba[0][1]*100:.2f}%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.write("### 📋 **Tabel Rangkuman Data Pasien**")
            df = pd.DataFrame(
            [
                {"waktu":current_time, "nama":nama, "umur":umur, "tinggi_badan":tinggi_badan, "berat_badan":berat_badan , "jenis_kelamin":jk, "sistolik":sistolik, "diastolik":diastolik, "kolesterol": kolesterol, "diabetes": diabetes, "riwayat":riwayat, "merokok": merokok, "olahraga":olahraga, "hasil": status}
            ]
            )
            # Styling warna berdasarkan hasil
            def highlight_hasil(val):
                if val == "Risiko Tinggi":
                    return "background-color: #ff4b4b; color: white;"
                elif val == "Risiko Rendah":
                    return "background-color: #28a745; color: white;"
                return ""

            styled_df = df.style.map(
                highlight_hasil,
                subset=["hasil"]
            )

            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )

# --- KODE FOOTER ---
footer = """
<style>
.footer {
    width: 100%;
    background-color: #f1f1f1;
    color: #333333;
    text-align: center;
    padding: 20px;
    font-size: 14px;
}
.footer p{
    font-size: 16px;
    text-align: center;
    margin-bottom:0;
}
</style>
<div class="footer">
    <p>Cardiovascular Care ❤️ Mencegah Lebih Baik Daripada Mengobati | © 2026</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
