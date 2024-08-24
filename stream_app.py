#Import Library
import streamlit as st
import numpy as np
import pandas as pd
import datetime
import time
from datetime import date, timedelta
from streamlit_option_menu import option_menu
#Import Library untuk Klasifikasi
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#Dapatkan waktu sekarang
current_time = datetime.datetime.now()
st.set_page_config(layout="wide")
# ----- Pengerjaan Model -----
data =pd.read_csv("Dataset.csv")
print(data.head())
#cleaning the data by dropping unneccessary column and dividing the data as features(x3) & target(y3)
X = data.drop(columns=['kardiovaskular'], axis=1)
Y = data['kardiovaskular']
scaler = StandardScaler()
scaler.fit(X)
data_standar = scaler.transform(X)
x=data_standar
y=Y
#performing train-test split on the data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
#creating an object for the model for further usage
classifier=RandomForestClassifier(n_estimators=100, random_state=1)
#fitting the model with train data (x3_train & y3_train)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(cm)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
# ------ Batas Pengerjaan Model -----

#Membuat Sidebar
with st.sidebar:
    col1, col2 = st.columns([1,2])
    with col1:
        st.image("image/logo-re.png", width=100)
    with col2:
        st.markdown('<h1 style="text-align:left">Rumah Sakit <br>dr. Suyoto</h1>', unsafe_allow_html=True)
    selected = option_menu("Main Menu", ["Beranda",'Informasi', 'Prediksi', 'Dataset', 'Visualisasi' , 'Tentang Kami'], 
        icons=['house', 'chat-heart', 'activity', 'clipboard-data', 'graph-up', 'person-circle'], menu_icon="cast", default_index=0)
if selected=='Informasi':
    st.header("Informasi Terkait Penyakit Kardiovaskular")
    st.html("<h3>Ancaman Penyakit Kardiovaskular</h3>")
    st.html("<p>Penyakit kardiovaskular masih menjadi ancaman dunia dan merupakan penyakit yang berperan utama sebagai penyebab kematian nomor satu di seluruh dunia. Data Organisasi Kesehatan Dunia (WHO) menyebutkan, lebih dari 17 juta orang di dunia meninggal akibat penyakit jantung dan pembuluh darah. Sedangkan sebagai perbandingan, HIV / AIDS, malaria dan TBC secara keseluruhan membunuh 3 juta populasi dunia. Berdasarkan data Riset Kesehatan Dasar (Riskesdas) tahun 2018, angka kejadian penyakit jantung dan pembuluh darah semakin meningkat dari tahun ke tahun. Setidaknya, 15 dari 1000 orang, atau sekitar 2.784.064 individu di Indonesia menderita penyakit jantung.</p>")
    st.html("<p>Penyakit kardiovaskular merupakan masalah kesehatan di negara maju maupun berkembang. Kementerian Kesehatan menyatakan, masyarakat perlu melakukan cek kesehatan berkala, menghindari perilaku merokok, rajin beraktivitas fisik, menerapkan pola makan seimbang, istirahat yang cukup, dan mengelola stres. Selain itu, masyarakat juga diimbau melakukan pengukuran tekanan darah dan rutin melakukan pemeriksaan kolesterol minimal satu tahun sekali.(Katadata)</p>")
    st.html("<h3>Fakta Penting</h3>")
    st.html("<style>h3{ margin-left: 50px} li, p{ font-size: 24px; text-align:justify; margin:8px 100px} ul li {background: #d4e9ff;;padding: 10px;border-radius: 10px;} ol li{background: #d7fcde;padding: 10px;border-radius: 10px;}  .faktor li{background: #fae4d9; padding: 10px;border-radius: 10px;} .faktor{margin-right:800px}</style>")
    st.html("<ul><li>Penyakit kardiovaskular merupakan penyebab kematian utama secara global.</li><li>Diperkirakan 17,9 juta orang meninggal akibat penyakit kardiovaskular pada tahun 2019, yang merupakan 32% dari seluruh kematian global. Dari jumlah tersebut, 85% disebabkan oleh serangan jantung dan stroke.</li><li>Lebih dari tiga perempat kematian akibat CVD terjadi di negara berpenghasilan rendah dan menengah.</li><li>Dari 17 juta kematian dini (di bawah usia 70) akibat penyakit tidak menular pada tahun 2019, 38% disebabkan oleh penyakit kardiovaskular.</li><li>Sebagian besar penyakit kardiovaskular dapat dicegah dengan menangani faktor risiko perilaku dan lingkungan seperti penggunaan tembakau, pola makan tidak sehat dan obesitas, kurangnya aktivitas fisik, penggunaan alkohol yang berbahaya, dan polusi udara.</li><li>Penting untuk mendeteksi penyakit kardiovaskular sedini mungkin sehingga penanganan dengan konseling dan pengobatan dapat dimulai.</li></ul> ")
    st.html("<h3>Jenis-jenis Penyakit Kardiovaskular</h3>")
    st.html("<ol><li><b>Jantung Koroner -</b> Penyakit jantung koroner terjadi ketika aliran darah kaya oksigen ke otot jantung tersumbat atau berkurang.</li><li><b>Stroke -</b> Stroke adalah kondisi saat suplai darah ke bagian otak terputus, yang dapat menyebabkan kerusakan otak dan kemungkinan kematian.</li><li><b>Aritmia -</b> Kondisi ini terjadi ketika detak jantung berlangsung dengan tidak teratur. Detak jantung bisa terjadi dengan sangat cepat atau sangat lambat.</li><li><b>Serangan Jantung -</b> Serangan jantung bisa terjadi akibat terputusnya aliran darah menuju otot jantung secara tiba-tiba.</li><li><b>Gagal Jantung -</b> Kondisi ini terjadi ketika jantung tidak mampu memompa darah untuk memenuhi kebutuhan tubuh.</li></ol> ")
    st.html("<h3>Faktor Risiko Penyebab Penyakit Kardiovaskular</h3>")
    st.html("<ol class='faktor'><li>Tekanan darah tinggi</li><li>Kolesterol tinggi</li><li>Diabetes</li><li>Obesitas(Berat badan berlebih)</li><li>Riwayat Keluarga yang pernah terkena kardiovaskular</li><li>Merokok</li><li>Kurangnya aktivitas fisik</li></ol> ")
    
if selected=='Dataset':
    st.subheader("Dataset Kardiovaskular")
    dataset = pd.read_csv('Dataset.csv')
    st.dataframe(dataset)
    st.download_button("Download Dataset", data='Dataset.csv', file_name="Dataset.csv", type='primary')

if selected=='Visualisasi':
    st.title(':chart_with_upwards_trend: Visualisasi Data ')
    st.header("1. Heatmap Correlation")
    st.image("image/Heatmap.png", caption="Heatmap Correlation Features")
#Membuat Halaman Home
if selected=='Beranda':
    st.title("Selamat Datang Di Website Cardiovascular Care :anatomical_heart:")
    st.html("<h3>Apa Saja Yang Kalian Bisa Lakukan di Website Ini? </h3>")
    st.html("")
    st.balloons()
    st.write("lorem "*200)

#Membuat Halaman Prediksi
if selected=='Prediksi':
    st.header(" :clipboard: Cek Risiko Kamu Terkena Penyakit Kardiovaskular")
    col1, col2, col3 = st.columns([2,1,1])
    col4, col5, col6, col7 = st.columns(4)
    jk = ("Perempuan", "Laki-laki")
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
        jenis_kelamin = st.selectbox("Jenis Kelamin", options, format_func=lambda x: jk[x])
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
    #Membuat Prediksi Pada Masukan
    input_data = (umur,jenis_kelamin,tinggi_badan,berat_badan,sistolik,diabetes,kolesterol,diabetes,riwayat,merokok,olahraga)
    print(input_data)
    input_data_as_numpy_array = np.array(input_data) 
    input_data_reshape = input_data_as_numpy_array.reshape(1,-1)
    std_data = scaler.transform(input_data_reshape)
    print(std_data)
    prediksi = classifier.predict(std_data)
    print(prediksi)
    status = ''
    if st.button("Prediksi", type="primary"):
        if(jenis_kelamin==1):
            jenis_kelamin = "Laki-laki"
        else:
            jenis_kelamin = "Perempuan"
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
        
        if (prediksi)==1:
            status = "Risiko Tinggi"
            with st.spinner('Sedang Memprediksi...'):
                time.sleep(3)
            st.subheader("Hasil Prediksi :")
            st.error("Kamu berisiko tinggi terkena penyakit kardiovaskular")
            df = pd.DataFrame(
            [
                {"waktu":current_time, "nama":nama, "umur":umur, "tinggi_badan":tinggi_badan, "berat_badan":berat_badan , "jenis_kelamin":jenis_kelamin, "kolesterol": kolesterol, "diabetes": diabetes, "riwayat":riwayat, "merokok": merokok, "olahraga":olahraga, "hasil": status}
            ]
            )
            st.dataframe(df, use_container_width=True)
        elif (prediksi)==0:
            if(nama==''):
                st.warning('Data belum diinput', icon="⚠️")
            else:
                status = "Risiko Rendah"
                with st.spinner('Sedang Memprediksi...'):
                    time.sleep(3)
                st.subheader("Hasil Prediksi :")
                st.success("Kamu berisiko rendah terkena penyakit kardiovaskular")
                df = pd.DataFrame(
                [
                    {"waktu":current_time, "nama":nama, "umur":umur, "tinggi_badan":tinggi_badan, "berat_badan":berat_badan , "jenis_kelamin":jenis_kelamin, "kolesterol": kolesterol, "diabetes": diabetes, "riwayat":riwayat, "merokok": merokok, "olahraga":olahraga, "hasil": status}
                ]
                )
                st.dataframe(df, use_container_width=True)
        

#Membuat Halaman About
if selected=='Tentang Kami':
    st.title("Selamat Datang Di Website Cardiovascular Care")
    st.write("This is About Page")
    st.balloons()
    