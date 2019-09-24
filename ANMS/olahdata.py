

#
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


## membuang strata selain S1
data_baru.drop(data_baru[data_baru['straamsjen'] != "S1"].index, inplace=True)
data_baru.head()


# In[19]:


data_baru.rename(index=str, columns={"nimhstrlsm":"NIM", "nmmhstrlsm":"nama", "tplhrtrlsm":"tempat lahir"
                                     , "tglhrtrlsm":"tgl lahir", "straamsjen":"strata"
                                     , "nmpstmspst":"jurusan", "llsketrlsm": "ket lulus", "nlipktrlsm":"ipk"
                                     , "tgllstrlsm":"tgl lulus", "tgmsktrlsm":"tgl masuk", "thstdtrlsm":"lama studi(tahun)"
                                     , "blstdtrlsm":"lama studi(bulan)", "hrstdtrlsm":"lama studi(hari)", "toefltrlsm":"toefl"
                                     , "noijftrlsm":"ijin(ragu)", "stwsdtrlsm":"stwsd(ragu)", "noijutrlsm":"noiju(ragu)"
                                     , "alamtrlsm":"alamat", "noteltrlsm":"telf rumah", "nohpetrlsm": "nomor hp"
                                     , "nmorttrlsm":"nama ortu", "skriptrlsm":"judul skripsi", "phototrlsm":"folder foto"
                                     , "jcutitrlsm":"cuti", "tgijatrlsm":"tgija(ragu)", "tahuntrlsm":"tahuntr(null)"
                                     , "preditrlsm":"predit(null)", "skrettrlsm":"skrektor(ragu)","tgrettrlsm":"tgl sk rektor"
                                     , "kdjektrlsm":"jenis kelamin", "dekanmsfak":"dekan fakultas", "nipnsmsfak":"nip dekan"
                                     , "nmfakmsfak":"fakultas", "kdfakmsfak":"kode fakultas", "pdek1msfak":"pemb dekan 1"
                                     , "nipn1msfak":"nip pemb 1", "nmtgamsjen":"skripsi/TA", "kdjenmsjen":"kdjen(ragu)"
                                     , "nmjenmsjen":"nama jenjang", "gelarmspst":"gelar", "kursitrlsm":"kursi(null)"
                                     , "pemb1trlsm":"pemb dekan 1 tr", "pemb2trlsm":"pemb dekan 2 tr", "kdpsttrlsm":"kodepsttr(ragu)"
                                     , "kdpstmspst":"kodepstm(ragu)","jalur":"jalur(ragu)"}, inplace=True)
data_baru.head()


# ** 3.2 Clean Data **
# - Pada proses Clean Data dilakukan dalam penanganan menghilangkan data kolom yang tidak digunakan ataupun memfilter data yang ada untuk diolah. Seperti data kolom 'nmmhstrlsm', 'tplhrtrlsm', dll.

# In[21]:


data_baru = data_baru.drop(['NIM','nama','lama studi(hari)','tempat lahir','ijin(ragu)'
                            ,'stwsd(ragu)','noiju(ragu)','telf rumah','nomor hp','nama ortu','folder foto'
                            ,'cuti','tgija(ragu)','tahuntr(null)','predit(null)','skrektor(ragu)'
                            ,'tgl sk rektor','dekan fakultas','nip dekan','fakultas','kode fakultas'
                            ,'pemb dekan 1','nip pemb 1','skripsi/TA','kdjen(ragu)','kursi(null)','gelar'
                            ,'nama jenjang','kodepsttr(ragu)','pemb dekan 2 tr','pemb dekan 1 tr'
                            ,'kodepstm(ragu)','jalur(ragu)','judul skripsi','tgl lahir', 'ket lulus'], axis=1)


# In[22]:


data_baru.head()


# ** 3.3 Construct Data **

# Membangun data agar data wisudawan yang dibuat dapat menghasilkan model yang sesuai dengan tujuan untuk menentukan faktor lulus tepat waktu/tidak tepat waktu

# In[23]:


data_pre = data_baru[['strata', 'jurusan', 'jenis kelamin', 'ipk', 'toefl', 
                     'tgl masuk', 'tgl lulus', 'alamat', 'lama studi(tahun)', 'lama studi(bulan)']]


# In[27]:


data_pre['strata'].value_counts()


# In[28]:


## Mengekstrak tanggal masuk untuk mengetahui angkatan
data_pre['tahun masuk'] = pd.DatetimeIndex(data_pre['tgl masuk']).year
data_pre.head(10)


# In[29]:


data_pre['lama studi(tahun)'].value_counts()


# In[30]:


data_pre['studi'] = (data_pre['lama studi(bulan)']/12)+data_pre['lama studi(tahun)']


# In[31]:


data_pre.head()


# In[32]:


data_pre.info()


# In[33]:


sns.relplot(x="toefl", y="ipk", hue="studi", data=data_pre);


# In[34]:


data_pre.loc[:, data_pre.all()]


# In[35]:


for i in range(data_pre.shape[0]):
    if data_pre['toefl'].iloc[i] < 200:
        print(data_pre['toefl'].iloc[i])


# In[36]:


data_pre.isnull().sum()


# In[37]:


data_pre.mean()


# Mengubah nilai 0 menjadi missing value untuk memudahkan dalam manipulasi nilai data

# In[38]:


data_tes = data_pre.replace([0],np.NaN)
data_tes.head()


# In[39]:


data_tes.mean()


# Berikut adalah jumlah data yang mempunyai nilai 0
# - TOEFL : 17 data
# - Studi : 1 data

# In[40]:


data_tes.isnull().sum()


# Mengisi nilai missing value dengan nilai rata-rata sesuai dengan kolom data yang digunakan

# In[41]:


data_tes['toefl'] = data_tes['toefl'].fillna(data_tes['toefl'].mean())
data_tes['studi'] = data_tes['studi'].fillna(data_tes['studi'].mean())


# In[42]:


data_tes.loc[:, data_tes.all()]


# Mengecek apakah nilai 0 yang berupa missing value sudah berubah dengan menggunakan nilai rata-rata kolom data yang sesuai

# In[43]:


data_tes.isnull().sum()


# Setelah melakukan analisis terhadap nilai data berupa 0, ternyata masih terdapat data yang tidak sesuai dengan ukuran nilai minimal dari kolom datanya
# - TOEFL minimal 200
# - Studi minimal < 3 tahun

# In[44]:


sns.relplot(x="toefl", y="ipk", hue="studi", data=data_tes);


# In[45]:


data_tes.studi.min()


# Masih terdapat nilai 40 pada kolom toefl, yang seharusnya toefl minimal 200

# In[46]:


data_v = data_tes.replace(40,np.NaN)
data_v


# In[47]:


data_v['toefl'] = data_v['toefl'].fillna(data_v['toefl'].mean())


# In[48]:


data_v.count()


# In[49]:


data_v.isnull().sum()


# In[50]:


sns.relplot(x="toefl", y="ipk", hue="studi", data=data_v);


# In[51]:


data_v.mean()


# In[52]:


data_v.toefl.min()


# Pada nilai studi masih terdapat nilai studi yang < 3 tahun, oleh karena itu nilai tsb diubah terlebih dahulu menjadi 0 untuk mempermudah pengubahan nilai

# In[53]:


for i in range(data_v.shape[0]):
    if data_v['studi'].iloc[i] < 3.0:
        print(data_v['studi'].iloc[i])
        data_v['studi'].iloc[i] = 0


# Mengubah nilai 0 pada studi menjadi missing value untuk mempermudah pengubahan nilai

# In[54]:


data_v1 = data_v.replace(0,np.NaN)
data_v1


# In[55]:


data_v1['studi'] = data_v1['studi'].fillna(data_v1['studi'].mean())


# Data kolom studi yang mempunyai nilai < 3 sudah berhasil dirubah menggunakan rata-rata studi

# In[56]:


data_v1.studi.min()


# In[57]:


data_v1.count()


# In[58]:


sns.relplot(x="toefl", y="ipk", hue="studi", data=data_v1);


# In[59]:


# writer = pd.ExcelWriter('DataNew.xlsx')
# data_v3.to_excel(writer, 'Sheet1', index=False)
# writer.save()


# ** 3.5 Format Data **
# - Melakukan tahap akhir dalam data preparation seperti mengubah tipe data, mengkategorikan data, ataupun yang berhubungan dengan persiapan data untuk diolah dalam modelling

# Mapping Lama Studi :
#     - Lama Studi > 3 dan <= 4 : 0 (Tepat Waktu)
#     - Lama Studi > 4 dan <= 6 : 1 (Tidak Tepat Waktu)
#     - Lama Studi > 6          : 2 (Rawan DO)

# In[60]:


# Mengkategorikan Lama Studi
data_v1.loc[ data_v1['studi'] <= 4, 'studi'] = 0,
data_v1.loc[ data_v1['studi'] > 4, 'studi'] = 1


# In[61]:


data_v1['studi'].value_counts()


# Melihat persentase masing-masing kategori abel data dari total data lama studi 

# In[62]:


tepat = 863
telat = 785
rawan = 163
total = 1811

print('Percent of Tepat: ''{0:.2f}%'.format((tepat / total * 100)))
print('Percent of Telat: ''{0:.2f}%'.format((telat / total * 100)))
print('Percent of Rawan: ''{0:.2f}%'.format((rawan / total * 100)))
print('Total Data: 1811')


# Melihat apakah kolom 'lama studi' sudah berhasil terbuat atau belum

# In[63]:


data_v1.head()


# In[64]:


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

# In[65]:


data_v1['ipk'].describe()


# In[66]:


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

# In[67]:


data_v1['toefl'].describe()


# In[68]:


# Mengkategorikan TOEFL
data_v1.loc[ data_v1['toefl'] <= 420, 'toefl'] = 0,
data_v1.loc[(data_v1['toefl'] > 420)& (data_v1['toefl'] <=480), 'toefl'] = 1,
data_v1.loc[(data_v1['toefl'] > 480)& (data_v1['toefl'] <=520), 'toefl'] = 2,
data_v1.loc[data_v1['toefl'] > 520, 'toefl'] = 3


# In[69]:


data_v1 = data_v1.drop(['lama studi(tahun)','lama studi(bulan)'], axis=1)


# In[70]:


data_v1.head(10)


# In[71]:


## Mengekstrak tanggal lulus untuk mengetahui tahun lulus
data_v1['tahun lulus'] = pd.DatetimeIndex(data_v1['tgl lulus']).year
data_v1.head()


# Mengkategorikan asal dari tiap wisudawan dengan kategori 'semarang' dan 'luar semarang'
#     
#     - Luar Semarang : 0
#     - Semarang      : 1

# In[72]:


data_v1['asal'] = data_v1['alamat'].str.contains('semarang', case=False, na=False).astype(int)


# In[73]:


data_v1.head()


# Mapping Jenis Kelamin :
#     - Laki-laki : 0
#     - Perempuan : 1

# In[74]:


# Mengkategorikan Jenis Kelamin
data_v1.loc[data_v1['jenis kelamin'] == 1, 'jenis kelamin'] = 0,
data_v1.loc[data_v1['jenis kelamin'] == 2, 'jenis kelamin'] = 1


# In[75]:


data_v1.head()


# Menghilangkan kolom data yang sudah diolah sedemikian rupa, hingga hanya menghasilkan beberapa kolom data

# In[76]:


data_v1 = data_v1.drop(['strata','tgl masuk','tgl lulus','alamat','tahun lulus'], axis=1)


# In[77]:


data_v1.head()


# Mengubah header kolom agar dalam pengolahan menjadi lebih mudah dipahami

# In[78]:


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

# In[79]:


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

# In[80]:


# Mengkategorikan Angkatan
data_v1.loc[data_v1['tahun masuk'] == 2005, 'tahun masuk'] = 0
data_v1.loc[data_v1['tahun masuk'] == 2006, 'tahun masuk'] = 1
data_v1.loc[data_v1['tahun masuk'] == 2007, 'tahun masuk'] = 2
data_v1.loc[data_v1['tahun masuk'] == 2008, 'tahun masuk'] = 3
data_v1.loc[data_v1['tahun masuk'] == 2009, 'tahun masuk'] = 4
data_v1.loc[data_v1['tahun masuk'] == 2010, 'tahun masuk'] = 5
data_v1.loc[data_v1['tahun masuk'] == 2011, 'tahun masuk'] = 6
data_v1.loc[data_v1['tahun masuk'] == 2012, 'tahun masuk'] = 7
data_v1.loc[data_v1['tahun masuk'] == 2013, 'tahun masuk'] = 8


# Merubah urutan header dengan urutan header pertama adalah 'lama studi' agar dalam tahap modelling dan deployment lebih mudah mengolahnya

# In[81]:


data_v1.head()


# In[82]:


def bar_chart(feature):
    tepat_waktu = data_v1[data_v1['studi']==0][feature].value_counts()
    tidak_tepat = data_v1[data_v1['studi']==1][feature].value_counts()
    df = pd.DataFrame([tepat_waktu,tidak_tepat])
    df.index = ['tepat_waktu','tidak_tepat']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[83]:


bar_chart('jenis kelamin')


# In[84]:


facet = sns.FacetGrid(data_v1, hue="studi",aspect=4)
facet.map(sns.kdeplot,'tahun masuk',shade= True)
facet.set(xlim=(0, data_v1['tahun masuk'].max()))
facet.add_legend()
plt.show() 


# In[85]:


data_final = data_v1[['jurusan', 'jenis kelamin', 'asal', 'ipk', 'toefl', 'tahun masuk', 'studi']]


# In[86]:


data_diskrit = data_v1[['jurusan', 'jenis kelamin', 'asal', 'ipk', 'toefl', 'studi']]


# In[87]:


# Mengkategorikan toefl
data_diskrit.loc[data_diskrit['toefl'] == 0, 'toefl'] = 'elementary'
data_diskrit.loc[data_diskrit['toefl'] == 1, 'toefl'] = 'low'
data_diskrit.loc[data_diskrit['toefl'] == 2, 'toefl'] = 'high'
data_diskrit.loc[data_diskrit['toefl'] == 3, 'toefl'] = 'advance'


# In[88]:


# Mengkategorikan IPK
data_diskrit.loc[data_diskrit['ipk'] == 0, 'ipk'] = 'cukup'
data_diskrit.loc[data_diskrit['ipk'] == 1, 'ipk'] = 'baik'
data_diskrit.loc[data_diskrit['ipk'] == 2, 'ipk'] = 'sangat_baik'
data_diskrit.loc[data_diskrit['ipk'] == 3, 'ipk'] = 'istimewa'


# In[89]:


# Mengkategorikan Jurusan
data_diskrit.loc[data_diskrit['jurusan'] == 0, 'jurusan'] = 'kim'
data_diskrit.loc[data_diskrit['jurusan'] == 1, 'jurusan'] = 'mat'
data_diskrit.loc[data_diskrit['jurusan'] == 2, 'jurusan'] = 'fis'
data_diskrit.loc[data_diskrit['jurusan'] == 3, 'jurusan'] = 'stat'
data_diskrit.loc[data_diskrit['jurusan'] == 4, 'jurusan'] = 'if'
data_diskrit.loc[data_diskrit['jurusan'] == 5, 'jurusan'] = 'bio'


# In[90]:


# Mengkategorikan jenis kelamin
data_diskrit.loc[data_diskrit['jenis kelamin'] == 0, 'jenis kelamin'] = 'L',
data_diskrit.loc[data_diskrit['jenis kelamin'] == 1, 'jenis kelamin'] = 'P'


# In[91]:


# Mengkategorikan asal
data_diskrit.loc[data_diskrit['asal'] == 0, 'asal'] = 'luar'
data_diskrit.loc[data_diskrit['asal'] == 1, 'asal'] = 'smg'


# In[92]:


# Mengkategorikan asal
data_diskrit.loc[data_diskrit['studi'] == 0, 'studi'] = 'tepat'
data_diskrit.loc[data_diskrit['studi'] == 1, 'studi'] = 'telat'


# In[93]:


from math import log

def entropy(*probs):
  """Calculate information entropy"""
  try:
    total = sum(probs)
    return sum([-p / total * log(p / total, 2) for p in probs])
  except:
    return 0


# In[94]:


entropy(1026, 785)


# In[95]:


# data_diskrit.loc[data_diskrit['ipk'] == "istimewa"]


# In[96]:


data_diskrit.head()


# In[97]:


data_diskrit.groupby(["ipk", "asal"]).size().reset_index(name="studi")


# In[98]:


studi_tepat = pd.crosstab(index=data_diskrit["studi"], 
                           columns=data_diskrit["jurusan"])

studi_tepat.index= ["telat","tepat"]

studi_tepat


# In[99]:


# #menyimpan dalam file excel
# writer = pd.ExcelWriter('DataDiskrit.xlsx')
# data_diskrit.to_excel(writer, 'Sheet1', index=False)
# writer.save()


# In[100]:


data_final.head()


# In[101]:


data_final = data_final.drop(['tahun masuk'], axis=1)


data_final.to_csv('static/datanumerik.csv', header=True, index=False)


# In[122]:


data_diskrit.to_csv('static/datadiskrit.csv', header=True, index=False)




