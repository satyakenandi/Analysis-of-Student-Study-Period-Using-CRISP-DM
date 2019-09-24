# get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import pickle
import warnings
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


# ** 3.1 Select Data **
# - Pada proses ini dilakukan untuk memilih data yang digunakan dan tidak digunakan. Karena di dalam tabel wisudawan masih banyak sekali tabel data yang tidak digunakan. Pada select data kali ini akan menghilangkan data yang selain S1

# In[3]:

data_baru = pd.read_csv('static/databaru.csv')


x = sns.relplot(x="toefltrlsm", y="nlipktrlsm", hue="thstdtrlsm", data=data_baru);
y = x.savefig("static/grafik.png")


def prepro():
    data_baru = pd.read_csv('static/databaru.csv')
    data_baru = data_baru.drop(["nimhstrlsm", "nmmhstrlsm", "tplhrtrlsm"
                                   , "tglhrtrlsm"
                                   , "llsketrlsm", "hrstdtrlsm"
                                   , "tgllstrlsm", "tgmsktrlsm"
                                   , "noijftrlsm", "stwsdtrlsm", "noijutrlsm"
                                   , "noteltrlsm", "nohpetrlsm"
                                   , "nmorttrlsm", "skriptrlsm", "phototrlsm"
                                   , "jcutitrlsm", "tgijatrlsm", "tahuntrlsm"
                                   , "preditrlsm", "skrettrlsm", "tgrettrlsm"
                                   , "dekanmsfak", "nipnsmsfak"
                                   , "nmfakmsfak", "kdfakmsfak", "pdek1msfak"
                                   , "nipn1msfak", "nmtgamsjen", "kdjenmsjen"
                                   , "nmjenmsjen", "gelarmspst", "kursitrlsm"
                                   , "pemb1trlsm", "pemb2trlsm", "kdpsttrlsm"
                                   , "kdpstmspst", "jalur"], axis=1)

    # In[10]:

    ## membuang strata selain S1
    data_baru = data_baru[data_baru['straamsjen'] == "S1"]
    data_baru

    # ** 3.3 Construct Data **

    # Membangun data agar data wisudawan yang dibuat dapat menghasilkan model yang sesuai dengan tujuan untuk menentukan faktor lulus tepat waktu/tidak tepat waktu

    # In[11]:

    data_baru.rename(index=str, columns={"straamsjen": "strata", "nmpstmspst": "jurusan", "nlipktrlsm": "ipk"
        , "toefltrlsm": "toefl", "kdjektrlsm": "jenis kelamin", "thstdtrlsm": "lama studi(tahun)"
        , "blstdtrlsm": "lama studi(bulan)", "alamtrlsm": "alamat"}, inplace=True)
    data_baru.describe()

    # In[12]:

    data_pre = data_baru[['strata', 'jurusan', 'jenis kelamin', 'ipk',
                          'toefl', 'alamat', 'lama studi(tahun)', 'lama studi(bulan)']]

    # In[13]:

    data_pre.describe()

    # In[14]:

    # #menyimpan dalam file excel
    # writer = pd.ExcelWriter('DataWisudawan.xlsx')
    # data_baru.to_excel(writer, 'Sheet1', index=False)
    # writer.save()

    # Melihat apakah data wisudawan sudah benar menghilangkan data wisudawan selain S1

    # In[15]:

    data_pre['strata'].value_counts()

    # Mengekstrak tanggal masuk dari kolom tgmsktrlsm untuk mengetahui angkatan mahasiswa dari tiap jurusan

    # Mengekstrak tanggal lulus dari kolom tgllstrlsm untuk mengetahui tahun lulus dari tiap mahasiswa

    # In[16]:

    ## Mengekstrak tanggal masuk untuk mengetahui angkatan
    # data_pre['tahun masuk'] = pd.DatetimeIndex(data_pre['tgl masuk']).year
    # data_pre.head(10)

    # ** 3.4 Integrate Data **
    # - Menggabungkan dua atau lebih tabel yang memiliki kemiripan informasi tentang objek yang sama ke dalam data baru.

    # Melihat data tahun studi untuk acuan menggabungkan data, dan didapatkan hasil :
    # - Lulus 4 Tahun = 885 orang
    # - Lulus 3 Tahun = 538 orang

    # In[17]:

    data_pre['lama studi(tahun)'].value_counts()

    # In[18]:

    data_pre['studi'] = (data_pre['lama studi(bulan)'] / 12) + data_pre['lama studi(tahun)']

    # In[19]:

    data_pre.head()

    # In[20]:

    data_pre.info()

    # In[21]:

    sns.relplot(x="toefl", y="ipk", hue="studi", data=data_pre);

    # Melihat kolom data yang tidak mempunyai nilai 0, dan yang mempunya nilai 0 adalah kolom :
    # - IPK
    # - TOEFL
    #
    # Sehingga nilai 0 tersebut harus direplace untuk meningkatkan akurasi

    # In[22]:

    data_pre.loc[:, data_pre.all()]

    # In[23]:

    for i in range(data_pre.shape[0]):
        if data_pre['toefl'].iloc[i] < 400:
            print(data_pre['toefl'].iloc[i])

    # In[24]:

    data_pre.isnull().sum()

    # In[25]:

    data_pre.mean()

    # Mengubah nilai 0 menjadi missing value untuk memudahkan dalam manipulasi nilai data

    # In[26]:

    data_tes = data_pre.replace([0], np.NaN)
    data_tes.head()

    # In[27]:

    data_tes.mean()

    # Berikut adalah jumlah data yang mempunyai nilai 0
    # - TOEFL : 17 data
    # - Studi : 1 data

    # In[28]:

    data_tes.isnull().sum()

    # Mengisi nilai missing value dengan nilai rata-rata sesuai dengan kolom data yang digunakan

    # In[29]:

    data_tes['toefl'] = data_tes['toefl'].fillna(data_tes['toefl'].mean())
    data_tes['studi'] = data_tes['studi'].fillna(data_tes['studi'].mean())

    # In[30]:

    data_tes.loc[:, data_tes.all()]

    # Mengecek apakah nilai 0 yang berupa missing value sudah berubah dengan menggunakan nilai rata-rata kolom data yang sesuai

    # In[31]:

    data_tes.isnull().sum()

    # Setelah melakukan analisis terhadap nilai data berupa 0, ternyata masih terdapat data yang tidak sesuai dengan ukuran nilai minimal dari kolom datanya
    # - TOEFL minimal 200
    # - Studi minimal < 3 tahun

    # In[32]:

    sns.relplot(x="toefl", y="ipk", hue="studi", data=data_tes);

    # In[33]:

    data_tes.studi.min()

    # Masih terdapat nilai 40 pada kolom toefl, yang seharusnya toefl minimal 200

    # In[34]:

    data_v = data_tes.replace(40, np.NaN)
    data_v

    # In[35]:

    data_v1 = data_v.replace(369, np.NaN)
    data_v1

    # In[36]:

    data_v1['toefl'] = data_v1['toefl'].fillna(data_v1['toefl'].mean())

    # In[37]:

    data_v1.count()

    # In[38]:

    data_v1.isnull().sum()

    # In[39]:

    sns.relplot(x="toefl", y="ipk", hue="studi", data=data_v1);

    # In[40]:

    data_v1.mean()

    # In[41]:

    data_v1.toefl.min()

    # Pada nilai studi masih terdapat nilai studi yang < 3 tahun, oleh karena itu nilai tsb diubah terlebih dahulu menjadi 0 untuk mempermudah pengubahan nilai

    # In[42]:

    for i in range(data_v1.shape[0]):
        if data_v1['studi'].iloc[i] < 3.0:
            print(data_v1['studi'].iloc[i])
            data_v1['studi'].iloc[i] = 0

    # Mengubah nilai 0 pada studi menjadi missing value untuk mempermudah pengubahan nilai

    # In[43]:

    data_v2 = data_v1.replace(0, np.NaN)
    data_v2

    # In[44]:

    data_v2['studi'] = data_v2['studi'].fillna(data_v2['studi'].mean())

    # Data kolom studi yang mempunyai nilai < 3 sudah berhasil dirubah menggunakan rata-rata studi

    # In[45]:

    data_v2.studi.min()

    # In[46]:

    data_v2.count()

    # In[47]:

    sns.relplot(x="toefl", y="ipk", hue="studi", data=data_v2);

    # In[48]:

    # writer = pd.ExcelWriter('DataNew.xlsx')
    # data_v3.to_excel(writer, 'Sheet1', index=False)
    # writer.save()

    # ** 3.5 Format Data **
    # - Melakukan tahap akhir dalam data preparation seperti mengubah tipe data, mengkategorikan data, ataupun yang berhubungan dengan persiapan data untuk diolah dalam modelling

    # Mapping Lama Studi :
    #     - Lama Studi > 3 dan <= 4 : 0 (Tepat Waktu)
    #     - Lama Studi > 4          : 1 (Tidak Tepat Waktu)

    # In[49]:

    # Mengkategorikan Lama Studi
    data_v2.loc[data_v2['studi'] <= 4, 'studi'] = 0,
    data_v2.loc[data_v2['studi'] > 4, 'studi'] = 1

    # In[50]:

    data_v2['studi'].value_counts()

    # Melihat persentase masing-masing kategori abel data dari total data lama studi

    # In[51]:

    tepat = 732
    telat = 1079
    total = 1811

    print('Percent of Tepat: ''{0:.2f}%'.format((tepat / total * 100)))
    print('Percent of Telat: ''{0:.2f}%'.format((telat / total * 100)))
    print('Total Data: 1811')

    # Melihat apakah kolom 'lama studi' sudah berhasil terbuat atau belum

    # In[52]:

    data_v2.head()

    # In[53]:

    sns.scatterplot(x=data_v2['toefl'], y=data_v2['ipk'], hue=data_v2['studi'])

    # Pengelompokan IPK akan dibagi menjadi 3 kategori,
    # yaitu tinggi, sangat memuaskan, dan rendah. Nilai per
    # kelompok dibagi berdasarkan Peraturan Rektor Universitas
    # Diponegoro Tahun 2012 Pasal 20.
    #
    # - IPK kurang dari 2,75 akan dikelompokkan sebagai IPK memuaskan,
    # - mahasiswa dengan IPK antara 2,76 sampai 3,50 akan dikelompokkan sebagai IPK sangat memuaskan,
    # - dan mahasiswa dengan IPK lebih dari 3,50 akan dikelompokkan sebagai IPK dengan pujian.
    #
    # Mapping IPK :
    #     - IPK <= 2.5 : 0
    #     - IPK > 2.5 dan <= 3.0 : 1
    #     - IPK > 3.0 dan <= 3.5 : 2
    #     - IPK > 3.5 dan <= 4.0 : 3

    # In[54]:

    data_v2['ipk'].describe()

    # In[55]:

    ## Mengkategorikan IPK
    data_v2.loc[data_v2['ipk'] <= 2.5, 'ipk'] = 0,
    data_v2.loc[(data_v2['ipk'] > 2.5) & (data_v2['ipk'] <= 3.0), 'ipk'] = 1,
    data_v2.loc[(data_v2['ipk'] > 3.0) & (data_v2['ipk'] <= 3.5), 'ipk'] = 2,
    data_v2.loc[data_v2['ipk'] > 3.5, 'ipk'] = 3

    # Dalam penelitian yang berjudul “Reading-Writing
    # Relationship in First and Second Language”, Carson dkk
    # mengelompokkan nilai TOEFL menjadi 4 kelas, yaitu
    # dasar, menengah bawah, menengah atas, dan mahir.
    #
    #
    #     - TOEFL < 420        = kelas dasar
    #     - TOEFL > 421 <= 480 = kelas menengah bawah
    #     - TOEFL > 481 <= 520 = kelas menengah atas
    #     - TOEFL > 520        = kelas mahir
    #
    # Mapping TOEFL :
    #     - TOEFL <= 420           : 0
    #     - TOEFL > 420 dan <= 480 : 1
    #     - TOEFL > 480 dan <= 520 : 2
    #     - TOEFL > 520            : 3

    # In[56]:

    data_v2['toefl'].describe()

    # In[57]:

    # Mengkategorikan TOEFL
    data_v2.loc[data_v2['toefl'] <= 420, 'toefl'] = 0,
    data_v2.loc[(data_v2['toefl'] > 420) & (data_v2['toefl'] <= 480), 'toefl'] = 1,
    data_v2.loc[(data_v2['toefl'] > 480) & (data_v2['toefl'] <= 520), 'toefl'] = 2,
    data_v2.loc[data_v2['toefl'] > 520, 'toefl'] = 3

    # In[58]:

    data_v2 = data_v2.drop(['lama studi(tahun)', 'lama studi(bulan)'], axis=1)

    # In[59]:

    data_v2.head(10)

    # In[60]:

    ## Mengekstrak tanggal lulus untuk mengetahui tahun lulus
    # data_v1['tahun lulus'] = pd.DatetimeIndex(data_v1['tgl lulus']).year
    # data_v1.head()

    # Mengkategorikan asal dari tiap wisudawan dengan kategori 'semarang' dan 'luar semarang'
    #
    #     - Luar Semarang : 0
    #     - Semarang      : 1

    # In[61]:

    data_v2['asal'] = data_v2['alamat'].str.contains('semarang', case=False, na=False).astype(int)

    # In[62]:

    data_v2.head()

    # Mapping Jenis Kelamin :
    #     - Laki-laki : 0
    #     - Perempuan : 1

    # In[63]:

    # Mengkategorikan Jenis Kelamin
    data_v2.loc[data_v2['jenis kelamin'] == 1, 'jenis kelamin'] = 0,
    data_v2.loc[data_v2['jenis kelamin'] == 2, 'jenis kelamin'] = 1

    # In[64]:

    data_v2.head()

    # Menghilangkan kolom data yang sudah diolah sedemikian rupa, hingga hanya menghasilkan beberapa kolom data

    # In[65]:

    data_v2 = data_v2.drop(['strata', 'alamat'], axis=1)

    # In[66]:

    data_v2.head()

    # Mengubah header kolom agar dalam pengolahan menjadi lebih mudah dipahami

    # In[67]:

    # data_v1.rename(index=str, columns={"nmpstmspst":"jurusan", "nlipktrlsm":"ipk",
    #                                      "toefltrlsm":"toefl","kdjektrlsm":"jenis kelamin"}, inplace=True)
    # data_v1.head()

    # Mapping Jurusan :
    #     - Kimia Murni   : 0
    #     - Matematika    : 1
    #     - Fisika        : 2
    #     - Statistika    : 3
    #     - Ilmu Komputer : 4
    #     - Biologi       : 5

    # In[68]:

    jurusan_mapping = {"Kimia Murni": 0, "Matematika": 1, "Fisika": 2, "Statistika": 3, "Ilmu Komputer": 4,
                       "Biologi": 5}
    data_v2['jurusan'] = data_v1['jurusan'].map(jurusan_mapping)

    # Mapping Angkatan :
    #     - Tahun 2005 : 0
    #     - Tahun 2006 : 1
    #     - Tahun 2007 : 2
    #     - Tahun 2008 : 3
    #     - Tahun 2009 : 4
    #     - Tahun 2010 : 5
    #     - Tahun 2011 : 6
    #     - Tahun 2012 : 7
    #     - Tahun 2013 : 8

    # In[69]:

    # # Mengkategorikan Angkatan
    # data_v1.loc[data_v1['tahun masuk'] == 2005, 'tahun masuk'] = 0
    # data_v1.loc[data_v1['tahun masuk'] == 2006, 'tahun masuk'] = 1
    # data_v1.loc[data_v1['tahun masuk'] == 2007, 'tahun masuk'] = 2
    # data_v1.loc[data_v1['tahun masuk'] == 2008, 'tahun masuk'] = 3
    # data_v1.loc[data_v1['tahun masuk'] == 2009, 'tahun masuk'] = 4
    # data_v1.loc[data_v1['tahun masuk'] == 2010, 'tahun masuk'] = 5
    # data_v1.loc[data_v1['tahun masuk'] == 2011, 'tahun masuk'] = 6
    # data_v1.loc[data_v1['tahun masuk'] == 2012, 'tahun masuk'] = 7
    # data_v1.loc[data_v1['tahun masuk'] == 2013, 'tahun masuk'] = 8

    # Merubah urutan header dengan urutan header pertama adalah 'lama studi' agar dalam tahap modelling dan deployment lebih mudah mengolahnya

    # In[70]:

    data_v2.head()

    # In[71]:

    # def bar_chart(feature):
    #     tepat_waktu = data_v1[data_v1['studi']==0][feature].value_counts()
    #     tidak_tepat = data_v1[data_v1['studi']==1][feature].value_counts()
    #     df = pd.DataFrame([tepat_waktu,tidak_tepat])
    #     df.index = ['tepat_waktu','tidak_tepat']
    #     df.plot(kind='bar',stacked=True, figsize=(10,5))

    # In[72]:

    # bar_chart('jenis kelamin')

    # In[73]:

    data_final = data_v2[['jurusan', 'jenis kelamin', 'asal', 'ipk', 'toefl', 'studi']]

    # In[74]:

    data_diskrit = data_v2[['jurusan', 'jenis kelamin', 'asal', 'ipk', 'toefl', 'studi']]

    # In[75]:

    # Mengkategorikan toefl
    data_diskrit.loc[data_diskrit['toefl'] == 0, 'toefl'] = 'elementary'
    data_diskrit.loc[data_diskrit['toefl'] == 1, 'toefl'] = 'low'
    data_diskrit.loc[data_diskrit['toefl'] == 2, 'toefl'] = 'high'
    data_diskrit.loc[data_diskrit['toefl'] == 3, 'toefl'] = 'advance'

    # In[76]:

    # Mengkategorikan IPK
    data_diskrit.loc[data_diskrit['ipk'] == 0, 'ipk'] = 'cukup'
    data_diskrit.loc[data_diskrit['ipk'] == 1, 'ipk'] = 'baik'
    data_diskrit.loc[data_diskrit['ipk'] == 2, 'ipk'] = 'sangat_baik'
    data_diskrit.loc[data_diskrit['ipk'] == 3, 'ipk'] = 'istimewa'

    # In[77]:

    # Mengkategorikan Jurusan
    data_diskrit.loc[data_diskrit['jurusan'] == 0, 'jurusan'] = 'kim'
    data_diskrit.loc[data_diskrit['jurusan'] == 1, 'jurusan'] = 'mat'
    data_diskrit.loc[data_diskrit['jurusan'] == 2, 'jurusan'] = 'fis'
    data_diskrit.loc[data_diskrit['jurusan'] == 3, 'jurusan'] = 'stat'
    data_diskrit.loc[data_diskrit['jurusan'] == 4, 'jurusan'] = 'if'
    data_diskrit.loc[data_diskrit['jurusan'] == 5, 'jurusan'] = 'bio'

    # In[78]:

    # Mengkategorikan jenis kelamin
    data_diskrit.loc[data_diskrit['jenis kelamin'] == 0, 'jenis kelamin'] = 'L',
    data_diskrit.loc[data_diskrit['jenis kelamin'] == 1, 'jenis kelamin'] = 'P'

    # In[79]:

    # Mengkategorikan asal
    data_diskrit.loc[data_diskrit['asal'] == 0, 'asal'] = 'luar'
    data_diskrit.loc[data_diskrit['asal'] == 1, 'asal'] = 'smg'

    # In[80]:

    # Mengkategorikan asal
    data_diskrit.loc[data_diskrit['studi'] == 0, 'studi'] = 'tepat'
    data_diskrit.loc[data_diskrit['studi'] == 1, 'studi'] = 'telat'

    # In[81]:

    from math import log

    def entropy(*probs):
        """Calculate information entropy"""
        try:
            total = sum(probs)
            return sum([-p / total * log(p / total, 2) for p in probs])
        except:
            return 0

    # In[82]:

    entropy(1026, 785)

    # In[83]:

    # data_diskrit.loc[data_diskrit['ipk'] == "istimewa"]

    # In[84]:

    data_diskrit.head()

    # In[85]:

    data_diskrit.groupby(["ipk", "asal"]).size().reset_index(name="studi")

    # In[86]:

    studi_tepat = pd.crosstab(index=data_diskrit["studi"],
                              columns=data_diskrit["jurusan"])

    studi_tepat.index = ["telat", "tepat"]

    studi_tepat

    # In[87]:

    # #menyimpan dalam file excel
    # writer = pd.ExcelWriter('DataDiskrit.xlsx')
    # data_diskrit.to_excel(writer, 'Sheet1', index=False)
    # writer.save()

    # In[88]:

    data_final.head()

    # In[89]:


    data_final.to_csv('static/datanumerik.csv', header=True, index=False)


    data_diskrit.to_csv('static/datadiskrit.csv', header=True, index=False)

    return data_final, data_diskrit



# Membersihkan data yang tidak diperlukan dan merubah header kolom sesuai dengan keperluan

# ## 4. Modelling 
# Modelling adalah pemilihan teknik data mining, algoritma dan menetukan parameter dengan nilai yang optimal. Pada tahapan pemodelan, ada beberapa hal yang dilakukan antara lain, memilih teknik pemodelan, membangun model, dan menilai model 

# ** 4.1 Select Modelling Technique **
# - Teknik data mining yang dipilih adalah decision tree. Decision tree sangat tepat digunakan untuk mencapai tujuan awal untuk mengetahui lulus tepat waktu/ tidak tepat waktu. Pemodelan data mining diawali dengan membuat rule untuk pembentukan pohon keputusan 

# ** 4.2 Asses Model **

# In[90]:



# In[102]:

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
def model():

    # import graphviz
    data_final, data_diskrit = prepro()
    features = list(data_final.columns[:5])
    features

    # In[92]:

    y = data_final["studi"]
    X = data_final[features]
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
    clf = clf.fit(X, y)

    data_final.head()

    # In[99]:

    X = data_final.drop(['studi'], axis=1)

    # In[100]:

    Y = data_final['studi']

    # In[101]:

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.40, random_state=40)

    # # 5. Evaluation (Scikit Learn)
    # In[103]:
    # data_hitung = pd.read_csv('static/datanumerik.csv')

    # ID3
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

    # In[104]:

    clf.fit(x_train, y_train)

    # In[105]:

    y_pred = clf.predict(x_test)

    # In[106]:

    metrics.accuracy_score(y_test, y_pred)

    # In[107]:

    metrics.precision_score(y_test, y_pred)

    # In[108]:

    metrics.recall_score(y_test, y_pred)

    # In[109]:

    metrics.confusion_matrix(y_test, y_pred)

    # In[110]:

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tn, fp, fn, tp

    # In[116]:

    # Akurasi
    akurasi = (tp + tn) / (tp + tn + fp + fn)
    akurasi

    # In[111]:

    sensitivity = tp / (tp + fn)
    sensitivity

    # In[112]:

    specifity = tn / (tn + fp)
    specifity

    # In[113]:

    # data_final.to_csv('datafix.csv', header=False, index=False)

    # In[114]:

    y_pred

    return akurasi,sensitivity,specifity


# ## 6. Deployment 

# Setelah didapatkan hasil melalui pengolahan data dengan label lama studi, dan menggunakan Decision Tree dalam menghitung evaluasi yang didapatkan yaitu 76%. Kemudian untuk mempermudah stakeholder dalam mengetahui informasi data maka dibuatlah sebuah web yang memberikan info mengenai studi kasus ini serta cara mengetahui data wisudawan yang lulus tepat waktu/tidak tepat waktu
# 
# Sehingga dengan adanya penilitian ini, pihak FSM Undip dapat meningkatkan upaya dan pengembangan agar lulusan yang ada di FSM Undip tiap tahunnya dapat lulus tepat waktu dan dengan IPK yang memuaskan.

# ## deploy to the Web
