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


data_baru = pd.read_csv('static/databaru.csv')


# In[13]:


# print(data_baru)


# In[14]:


sns.relplot(x="kdjektrlsm", y="nlipktrlsm", hue="thstdtrlsm", data=data_baru);


# In[15]:


#menyimpan dalam file excel
# writer = pd.ExcelWriter('DataBaru.xlsx')
# data_baru.to_excel(writer, 'Sheet1', index=False)
# writer.save()


# In[16]:


data_baru.isnull().sum()


# Setelah menggabungkan semua data excel, kemudian data excel yang telah tergabung dilihat apakah terdapat missing value atau tidak di dalam kolomnya dari total 2739 data. 
# Hasil sementara kolom yang terdapat missing value :
# - noijftrlsm : 2739 data
# - alamtrlsm  : 1 data
# - telf rumah : 1807 data
# - nomor hp   : 5 data
# - nama ortu  : 2 data
# - tahuntrlsm : 366 data
# - skrektor   : 1810 data
# - kursitrlsm : 668 data
# - pemb 1 tr  : 4 data
# - pemb 2 t   : 375 data

# ** 3.2 Clean Data **
# - Pada proses Clean Data dilakukan dalam penanganan menghilangkan data kolom yang tidak digunakan ataupun memfilter data yang ada untuk diolah. Seperti data kolom 'nmmhstrlsm', 'tplhrtrlsm', dll.

# In[17]:


data_baru = data_baru.drop(["nimhstrlsm", "nmmhstrlsm", "tplhrtrlsm"
                            , "tglhrtrlsm"
                            , "llsketrlsm", "hrstdtrlsm"
                            , "tgllstrlsm", "tgmsktrlsm"
                            , "noijftrlsm", "stwsdtrlsm", "noijutrlsm"
                            , "noteltrlsm", "nohpetrlsm"
                            , "nmorttrlsm", "skriptrlsm", "phototrlsm"
                            , "jcutitrlsm", "tgijatrlsm", "tahuntrlsm"
                            , "preditrlsm", "skrettrlsm","tgrettrlsm"
                            , "dekanmsfak", "nipnsmsfak"
                            , "nmfakmsfak", "kdfakmsfak", "pdek1msfak"
                            , "nipn1msfak", "nmtgamsjen", "kdjenmsjen"
                            , "nmjenmsjen", "gelarmspst", "kursitrlsm"
                            , "pemb1trlsm", "pemb2trlsm", "kdpsttrlsm"
                            , "kdpstmspst","jalur"], axis=1)


# In[18]:


## membuang strata selain S1
data_baru = data_baru[data_baru['straamsjen'] == "S1"]
data_baru.head()


# ** 3.3 Construct Data **

# Membangun data agar data wisudawan yang dibuat dapat menghasilkan model yang sesuai dengan tujuan untuk menentukan faktor lulus tepat waktu/tidak tepat waktu

# In[19]:


data_baru.rename(index=str, columns={"straamsjen":"strata", "nmpstmspst":"jurusan", "nlipktrlsm":"ipk"
                                     , "toefltrlsm":"toefl", "kdjektrlsm":"jenis kelamin", "thstdtrlsm":"lama studi(tahun)"
                                     , "blstdtrlsm":"lama studi(bulan)", "alamtrlsm":"alamat"}, inplace=True)
data_baru.head()

print(data_baru)


# In[20]:


data_pre = data_baru[['strata', 'jurusan', 'jenis kelamin', 'ipk', 
                      'toefl', 'alamat', 'lama studi(tahun)', 'lama studi(bulan)']]


# In[21]:


data_pre.head()


# In[22]:


# #menyimpan dalam file excel
# writer = pd.ExcelWriter('DataWisudawan.xlsx')
# data_baru.to_excel(writer, 'Sheet1', index=False)
# writer.save()


# Melihat apakah data wisudawan sudah benar menghilangkan data wisudawan selain S1

# In[23]:


data_pre['strata'].value_counts()


# Mengekstrak tanggal masuk dari kolom tgmsktrlsm untuk mengetahui angkatan mahasiswa dari tiap jurusan

# Mengekstrak tanggal lulus dari kolom tgllstrlsm untuk mengetahui tahun lulus dari tiap mahasiswa

# In[24]:


## Mengekstrak tanggal masuk untuk mengetahui angkatan
# data_pre['tahun masuk'] = pd.DatetimeIndex(data_pre['tgl masuk']).year
# data_pre.head(10)


# ** 3.4 Integrate Data **
# - Menggabungkan dua atau lebih tabel yang memiliki kemiripan informasi tentang objek yang sama ke dalam data baru.

# Melihat data tahun studi untuk acuan menggabungkan data, dan didapatkan hasil :
# - Lulus 4 Tahun = 885 orang
# - Lulus 3 Tahun = 538 orang

# In[25]:


data_pre['lama studi(tahun)'].value_counts()


# In[26]:


data_pre['studi'] = (data_pre['lama studi(bulan)']/12)+data_pre['lama studi(tahun)']


# In[27]:


data_pre.head()


# In[28]:


data_pre.info()


# In[29]:


sns.relplot(x="toefl", y="ipk", hue="studi", data=data_pre);


# Melihat kolom data yang tidak mempunyai nilai 0, dan yang mempunya nilai 0 adalah kolom :
# - IPK
# - TOEFL
# 
# Sehingga nilai 0 tersebut harus direplace untuk meningkatkan akurasi 

# In[30]:


data_pre.loc[:, data_pre.all()]


# In[31]:


for i in range(data_pre.shape[0]):
    if data_pre['toefl'].iloc[i] < 200:
        print(data_pre['toefl'].iloc[i])


# In[32]:


data_pre.isnull().sum()


# In[33]:


data_pre.mean()


# Mengubah nilai 0 menjadi missing value untuk memudahkan dalam manipulasi nilai data

# In[34]:


data_tes = data_pre.replace([0],np.NaN)
data_tes.head()


# In[35]:


data_tes.mean()


# Berikut adalah jumlah data yang mempunyai nilai 0
# - TOEFL : 17 data
# - Studi : 1 data

# In[36]:


data_tes.isnull().sum()


# Mengisi nilai missing value dengan nilai rata-rata sesuai dengan kolom data yang digunakan

# In[37]:


data_tes['toefl'] = data_tes['toefl'].fillna(data_tes['toefl'].mean())
data_tes['studi'] = data_tes['studi'].fillna(data_tes['studi'].mean())


# In[38]:


data_tes.loc[:, data_tes.all()]


# Mengecek apakah nilai 0 yang berupa missing value sudah berubah dengan menggunakan nilai rata-rata kolom data yang sesuai

# In[39]:


data_tes.isnull().sum()


# Setelah melakukan analisis terhadap nilai data berupa 0, ternyata masih terdapat data yang tidak sesuai dengan ukuran nilai minimal dari kolom datanya
# - TOEFL minimal 200
# - Studi minimal < 3 tahun

# In[40]:


sns.relplot(x="toefl", y="ipk", hue="studi", data=data_tes);


# In[41]:


data_tes.studi.min()


# Masih terdapat nilai 40 pada kolom toefl, yang seharusnya toefl minimal 200

# In[42]:


data_v = data_tes.replace(40,np.NaN)
data_v


# In[43]:


data_v['toefl'] = data_v['toefl'].fillna(data_v['toefl'].mean())


# In[44]:


data_v.count()


# In[45]:


data_v.isnull().sum()


# In[46]:


sns.relplot(x="toefl", y="ipk", hue="studi", data=data_v);


# In[47]:


data_v.mean()


# In[48]:


data_v.toefl.min()


# Pada nilai studi masih terdapat nilai studi yang < 3 tahun, oleh karena itu nilai tsb diubah terlebih dahulu menjadi 0 untuk mempermudah pengubahan nilai

# In[49]:


for i in range(data_v.shape[0]):
    if data_v['studi'].iloc[i] < 3.0:
        print(data_v['studi'].iloc[i])
        data_v['studi'].iloc[i] = 0


# Mengubah nilai 0 pada studi menjadi missing value untuk mempermudah pengubahan nilai

# In[50]:


data_v1 = data_v.replace(0,np.NaN)
data_v1


# In[51]:


data_v1['studi'] = data_v1['studi'].fillna(data_v1['studi'].mean())


# Data kolom studi yang mempunyai nilai < 3 sudah berhasil dirubah menggunakan rata-rata studi

# In[52]:


data_v1.studi.min()


# In[53]:


data_v1.count()


# In[54]:


sns.relplot(x="toefl", y="ipk", hue="studi", data=data_v1);


# In[55]:


# writer = pd.ExcelWriter('DataNew.xlsx')
# data_v3.to_excel(writer, 'Sheet1', index=False)
# writer.save()


# ** 3.5 Format Data **
# - Melakukan tahap akhir dalam data preparation seperti mengubah tipe data, mengkategorikan data, ataupun yang berhubungan dengan persiapan data untuk diolah dalam modelling

# Mapping Lama Studi :
#     - Lama Studi > 3 dan <= 4 : 0 (Tepat Waktu)
#     - Lama Studi > 4          : 1 (Tidak Tepat Waktu)

# In[56]:


# Mengkategorikan Lama Studi
data_v1.loc[ data_v1['studi'] <= 4, 'studi'] = 0,
data_v1.loc[ data_v1['studi'] > 4, 'studi'] = 1


# In[57]:


data_v1['studi'].value_counts()


# Melihat persentase masing-masing kategori abel data dari total data lama studi 

# In[58]:


tepat = 732
telat = 1079
total = 1811

print('Percent of Tepat: ''{0:.2f}%'.format((tepat / total * 100)))
print('Percent of Telat: ''{0:.2f}%'.format((telat / total * 100)))
print('Total Data: 1811')


# Melihat apakah kolom 'lama studi' sudah berhasil terbuat atau belum

# In[59]:


data_v1.head()


# In[60]:


sns.scatterplot(x=data_v1['toefl'], y=data_v1['ipk'], hue=data_v1['studi'])


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

# In[61]:


data_v1['ipk'].describe()


# In[62]:


## Mengkategorikan IPK 
data_v1.loc[ data_v1['ipk'] <= 2.5, 'ipk'] = 0,
data_v1.loc[(data_v1['ipk'] > 2.5)& (data_v1['ipk'] <=3.0), 'ipk'] = 1,
data_v1.loc[(data_v1['ipk'] > 3.0)& (data_v1['ipk'] <=3.5), 'ipk'] = 2,
data_v1.loc[ data_v1['ipk'] > 3.5, 'ipk'] = 3


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

# In[63]:


data_v1['toefl'].describe()


# In[64]:


# Mengkategorikan TOEFL
data_v1.loc[ data_v1['toefl'] <= 420, 'toefl'] = 0,
data_v1.loc[(data_v1['toefl'] > 420)& (data_v1['toefl'] <=480), 'toefl'] = 1,
data_v1.loc[(data_v1['toefl'] > 480)& (data_v1['toefl'] <=520), 'toefl'] = 2,
data_v1.loc[data_v1['toefl'] > 520, 'toefl'] = 3


# In[65]:


data_v1 = data_v1.drop(['lama studi(tahun)','lama studi(bulan)'], axis=1)


# In[66]:


data_v1.head(10)


# In[67]:


## Mengekstrak tanggal lulus untuk mengetahui tahun lulus
# data_v1['tahun lulus'] = pd.DatetimeIndex(data_v1['tgl lulus']).year
# data_v1.head()


# Mengkategorikan asal dari tiap wisudawan dengan kategori 'semarang' dan 'luar semarang'
#     
#     - Luar Semarang : 0
#     - Semarang      : 1

# In[68]:


data_v1['asal'] = data_v1['alamat'].str.contains('semarang', case=False, na=False).astype(int)


# In[69]:


data_v1.head()


# Mapping Jenis Kelamin :
#     - Laki-laki : 0
#     - Perempuan : 1

# In[70]:


# Mengkategorikan Jenis Kelamin
data_v1.loc[data_v1['jenis kelamin'] == 1, 'jenis kelamin'] = 0,
data_v1.loc[data_v1['jenis kelamin'] == 2, 'jenis kelamin'] = 1


# In[71]:


data_v1.head()


# Menghilangkan kolom data yang sudah diolah sedemikian rupa, hingga hanya menghasilkan beberapa kolom data

# In[72]:


data_v1 = data_v1.drop(['alamat'], axis=1)


# In[73]:


data_v1.head()


# Mengubah header kolom agar dalam pengolahan menjadi lebih mudah dipahami

# In[74]:


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

# In[75]:


jurusan_mapping = {"Kimia Murni": 0, "Matematika":1, "Fisika":2, "Statistika":3, "Ilmu Komputer":4, "Biologi":5}
data_v1['jurusan'] = data_v1['jurusan'].map(jurusan_mapping)


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

# In[76]:


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

# In[77]:


data_v1.head()


# In[78]:


def bar_chart(feature):
    tepat_waktu = data_v1[data_v1['studi']==0][feature].value_counts()
    tidak_tepat = data_v1[data_v1['studi']==1][feature].value_counts()
    df = pd.DataFrame([tepat_waktu,tidak_tepat])
    df.index = ['tepat_waktu','tidak_tepat']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[79]:


bar_chart('jenis kelamin')


# In[80]:


data_final = data_v1[['jurusan', 'jenis kelamin', 'asal', 'ipk', 'toefl', 'studi']]


# In[81]:


data_diskrit = data_v1[['strata','jurusan', 'jenis kelamin', 'asal', 'ipk', 'toefl', 'studi']]


# In[82]:


# Mengkategorikan toefl
data_diskrit.loc[data_diskrit['toefl'] == 0, 'toefl'] = 'elementary'
data_diskrit.loc[data_diskrit['toefl'] == 1, 'toefl'] = 'low'
data_diskrit.loc[data_diskrit['toefl'] == 2, 'toefl'] = 'high'
data_diskrit.loc[data_diskrit['toefl'] == 3, 'toefl'] = 'advance'


# In[83]:


# Mengkategorikan IPK
data_diskrit.loc[data_diskrit['ipk'] == 0, 'ipk'] = 'cukup'
data_diskrit.loc[data_diskrit['ipk'] == 1, 'ipk'] = 'baik'
data_diskrit.loc[data_diskrit['ipk'] == 2, 'ipk'] = 'sangat_baik'
data_diskrit.loc[data_diskrit['ipk'] == 3, 'ipk'] = 'istimewa'


# In[84]:


# Mengkategorikan Jurusan
data_diskrit.loc[data_diskrit['jurusan'] == 0, 'jurusan'] = 'kim'
data_diskrit.loc[data_diskrit['jurusan'] == 1, 'jurusan'] = 'mat'
data_diskrit.loc[data_diskrit['jurusan'] == 2, 'jurusan'] = 'fis'
data_diskrit.loc[data_diskrit['jurusan'] == 3, 'jurusan'] = 'stat'
data_diskrit.loc[data_diskrit['jurusan'] == 4, 'jurusan'] = 'if'
data_diskrit.loc[data_diskrit['jurusan'] == 5, 'jurusan'] = 'bio'


# In[85]:


# Mengkategorikan jenis kelamin
data_diskrit.loc[data_diskrit['jenis kelamin'] == 0, 'jenis kelamin'] = 'L',
data_diskrit.loc[data_diskrit['jenis kelamin'] == 1, 'jenis kelamin'] = 'P'


# In[86]:


# Mengkategorikan asal
data_diskrit.loc[data_diskrit['asal'] == 0, 'asal'] = 'luar'
data_diskrit.loc[data_diskrit['asal'] == 1, 'asal'] = 'smg'


# In[87]:


# Mengkategorikan asal
data_diskrit.loc[data_diskrit['studi'] == 0, 'studi'] = 'tepat'
data_diskrit.loc[data_diskrit['studi'] == 1, 'studi'] = 'telat'


# In[88]:


from math import log

def entropy(*probs):
  """Calculate information entropy"""
  try:
    total = sum(probs)
    return sum([-p / total * log(p / total, 2) for p in probs])
  except:
    return 0


# In[89]:


entropy(1026, 785)


# In[90]:


# data_diskrit.loc[data_diskrit['ipk'] == "istimewa"]


# In[91]:


data_diskrit.head()


# In[92]:


data_diskrit.groupby(["ipk", "asal"]).size().reset_index(name="studi")


# In[93]:


studi_tepat = pd.crosstab(index=data_diskrit["studi"], 
                           columns=data_diskrit["jurusan"])

studi_tepat.index= ["telat","tepat"]

studi_tepat


# In[94]:


# #menyimpan dalam file excel
# writer = pd.ExcelWriter('DataDiskrit.xlsx')
# data_diskrit.to_excel(writer, 'Sheet1', index=False)
# writer.save()


# In[95]:


data_final.head()

data_final.to_csv('static/datanumerik.csv', header=True, index=False)


# In[122]:


data_diskrit.to_csv('static/datadiskrit.csv', header=True, index=False)

# In[96]:


# data_final.to_csv('datafix.csv', header=True, index=False)


# Membersihkan data yang tidak diperlukan dan merubah header kolom sesuai dengan keperluan

# ## 4. Modelling 
# Modelling adalah pemilihan teknik data mining, algoritma dan menetukan parameter dengan nilai yang optimal. Pada tahapan pemodelan, ada beberapa hal yang dilakukan antara lain, memilih teknik pemodelan, membangun model, dan menilai model 

# ** 4.1 Select Modelling Technique **
# - Teknik data mining yang dipilih adalah decision tree. Decision tree sangat tepat digunakan untuk mencapai tujuan awal untuk mengetahui lulus tepat waktu/ tidak tepat waktu. Pemodelan data mining diawali dengan membuat rule untuk pembentukan pohon keputusan 

# ** 4.2 Asses Model **

# In[97]:

#
# from sklearn.model_selection import train_test_split
# from sklearn import tree
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
#
#
# features = list(data_final.columns[:5])
# features
#
#
# # In[99]:
#
#
# y = data_final["studi"]
# X = data_final[features]
# clf = tree.DecisionTreeClassifier(criterion = "entropy", max_depth=3, random_state=0)
# clf = clf.fit(X,y)
#
#
# data_final.head()
#
#
# # In[106]:
#
#
# X = data_final.drop(['studi'], axis=1)
#
#
# # In[107]:
#
#
# Y = data_final['studi']
#
#
# # In[108]:
#
#
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.40, random_state=40)
#
#
# # # 5. Evaluation (Scikit Learn)
#
# # In[109]:
#
#
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import metrics
# from sklearn.externals.six import StringIO
#
#
# # In[110]:
#
#
# # ID3
# clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
#
#
# # In[111]:
#
#
# clf.fit(x_train, y_train)
#
#
# # In[112]:
#
#
# y_pred = clf.predict(x_test)
#
#
# # In[113]:
#
#
# metrics.accuracy_score(y_test, y_pred)
#
#
# # In[114]:
#
#
# metrics.precision_score(y_test, y_pred)
#
#
# # In[115]:
#
#
# metrics.recall_score(y_test, y_pred)
#
#
# # In[116]:
#
#
# metrics.confusion_matrix(y_test, y_pred)
#
#
# # In[117]:
#
#
# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# tn, fp, fn, tp
#
#
# # In[118]:
#
#
# tp / (tp + fn)
#
#
# # In[119]:
#
#
# tn / (tn + fp)
#
#
# # In[120]:
#
#
# # data_final.to_csv('datafix.csv', header=False, index=False)
#
#
# # In[121]:
#
#
# y_pred

