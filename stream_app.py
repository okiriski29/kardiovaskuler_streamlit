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
# ----- Pengerjaan Model -----
data =pd.read_csv("Dataset.csv")
print(data.head())
#cleaning the data by dropping unneccessary column and dividing the data as features(x3) & target(y3)
X = data.drop(columns=['kardiovaskular'], axis=1)
Y = data['kardiovaskular']
scaler = StandardScaler()
scaler.fit(X.values)
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
    st.image("logo-re.png", width=100)
    selected = option_menu("Main Menu", ["Beranda", 'Prediksi', 'Tentang Kami'], 
        icons=['house', 'activity', 'info-circle'], menu_icon="cast", default_index=0)
#Membuat Halaman Home
if selected=='Beranda':
    st.title("Selamat Datang Di Website Cardiovascular Care")
    st.write("This is Homepage")
    st.balloons()

#Membuat Halaman Prediksi
if selected=='Prediksi':
    st.header("Cek Risiko Kamu Terkena Penyakit Kardiovaskular")
    col1, col2 = st.columns([2,1])
    col3, col4, col5 = st.columns(3)
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
        berat_badan=st.number_input("Berat Badan",value=None, min_value=40,max_value=150,step=1)
        kolesterol = st.selectbox("Kolesterol", options, format_func=lambda x: option[x])
        merokok = st.selectbox("Merokok", options, format_func=lambda x: option[x])
    with col4:
        jenis_kelamin = st.selectbox("Jenis Kelamin", options, format_func=lambda x: jk[x])
        sistolik = st.number_input("Tekanan Sistolik",value=None, min_value=70,max_value=250,step=10)
        diabetes = st.selectbox("Diabetes", options, format_func=lambda x: option[x])
        olahraga = st.selectbox("Olahraga", options, format_func=lambda x: option[x])
    with col5:
        tinggi_badan = st.number_input("Tinggi Badan",value=None, min_value=125,max_value=565,step=1)
        diastolik = st.number_input("Tekanan Diastolik",value=None, min_value=40,max_value=160,step=10)
        riwayat = st.selectbox("Riwayat Keluarga", options, format_func=lambda x: option[x])
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
    if st.button("Predict", type="primary"):
        st.header("Hasil Prediksi:")
        if(jenis_kelamin==1):
            jenis_kelamin = "Laki-laki"
        else:
            jenis_kelamin = "Perempuan"
        if (prediksi)==1:
            status = "Risiko Tinggi"
            st.warning("Kamu berisiko tinggi terkena penyakit kardiovaskular")
            df = pd.DataFrame(
            [
                {"waktu":current_time, "nama":nama, "umur":umur, "tinggi_badan":tinggi_badan, "berat_badan":berat_badan , "jenis_kelamin":jenis_kelamin, "kolesterol": kolesterol, "diabetes": diabetes, "merokok": merokok, "hasil": status}
            ]
            )
            st.dataframe(df, use_container_width=True)
        elif (prediksi)==0:
            if(nama==''):
                st.warning('Data belum diinput', icon="⚠️")
            else:
                status = "Risiko Rendah"
                st.success("Kamu berisiko rendah terkena penyakit kardiovaskular")
                df = pd.DataFrame(
                [
                    {"waktu":current_time, "nama":nama, "umur":umur, "tinggi_badan":tinggi_badan, "berat_badan":berat_badan , "jenis_kelamin":jenis_kelamin, "kolesterol": kolesterol, "diabetes": diabetes, "merokok": merokok, "hasil": status}
                ]
                )
                st.dataframe(df, use_container_width=True)
        

#Membuat Halaman About
if selected=='Tentang Kami':
    st.title("Selamat Datang Di Website Cardiovascular Care")
    st.write("This is About Page")
    st.balloons()
    st.image("image/Heatmap.png", caption="Heatmap Correlation Features")