import base64
import io
# from PIL import Image

from flask import Flask, render_template, request, redirect, flash
from sklearn.tree import DecisionTreeClassifier
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from markupsafe import Markup
import os
import pickle as pk

# from olahdata import *
# from hasildata import *
from preprocessing import *
from tree import *
from treefix import *

app = Flask(__name__)
app.secret_key = 'sabeb'

dtc_model = DecisionTreeClassifier(criterion="entropy")

DEFAULT_JURUSAN = 'Kimia'
DEFAULT_IPK = '<= 2.5'
DEFAULT_TOEFL = '<= 420'
DEFAULT_JENKEL = 'Laki - laki'
DEFAULT_ASAL = 'Semarang'


@app.route('/')
def main():
    initree()
    # hasil_akurasi = akurasi
    # hasil_sensitivity = sensitivity
    # hasil_specifity = specifity
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # hasil_akurasi = akurasi
    # hasil_sensitivity = sensitivity
    # hasil_specifity = specifity
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/visualize')
def visualize():
    data_dict = json.load(open("static/converted_tree.json"))
    return render_template('visualize.html', data={"tree":data_dict})

@app.route('/infotree')
def infotree():
    return render_template('infotree.html')

@app.route('/infotabel')
def infotabel():
    return render_template('infotabel.html')

@app.route('/proses', methods = ['GET', 'POST'])
def proses():
    if request.method == 'POST':
        f = request.files['file']
        f.save('static/' + secure_filename("databaru.csv"))
        os.system('python preprocessing.py')
        os.system("python treefix.py")
        # import preprocessing
        # import treefix
        msg="Data Berhasil di upload"
    else:
        msg="Data Gagal di upload"
    return render_template("upload.html",messg=msg)


@app.route('/visualisasi')
def visualisasi():
    return render_template('visualisasi.html')

data_final, data_diskrit=(None, None)

@app.route('/tabel')
def tabel():
    data_final, data_diskrit = prepro()
    initree()
    tabel = data_diskrit
    # tabel = pd.read_csv("static/datadiskrit.csv")
    kolom = list(tabel)
    hasil = ""
    header = "<th>Nomor</th>"

    for x in range(len(kolom)):
        isi = str(kolom[x])
        header = header + "<th>" + str(isi) + "</th>"

    for x in range(len(tabel.index)):
        hasil = hasil + "<tr>"
        hasil = hasil + "<td>" + str(x + 1) + "</td>"
        for y in range(len(kolom)):
            hasil = hasil + "<td>" + str(tabel[kolom[y]][x]) + "</td>"
        hasil = hasil + "</tr>"
    return render_template('tabel.html', header=Markup(header), record=Markup(hasil))


# @app.route('/awal')
# def awal():
#     tabel = data_awal
#     kolom = list(tabel)
#     hasil = ""
#     header = "<th>Nomor</th>"/
#
#     for x in range(len(kolom)):
#         isi = str(kolom[x])
#         header = header + "<th>" + str(isi) + "</th>"
#
#     for x in range(len(tabel.index)):
#         hasil = hasil + "<tr>"
#         hasil = hasil + "<td>" + str(x + 1) + "</td>"
#         for y in range(len(kolom)):
#             hasil = hasil + "<td>" + str(tabel[kolom[y]][x]) + "</td>"
#         hasil = hasil + "</tr>"
#     return render_template('awal.html', header=Markup(header), record=Markup(hasil))

@app.route('/tree')
def tree():
    list_tree = ""
    for i in cek:
        list_tree = list_tree + str(i) + "&#13;&#10;"
    return render_template('tree.html', cek=Markup(list_tree))


@app.before_first_request
def startup():
    predict()

def predict():
    data_final, data_diskrit = prepro()
    global dtc_model

    from numpy import genfromtxt
    wisuda_array = genfromtxt('static/datanumerik.csv', delimiter=',')
    # print(wisuda_array)

    X_train, X_test, y_train, y_test = train_test_split([item[1:] for item in wisuda_array],
                                                        [item[0] for item in wisuda_array], test_size=0.4,
                                                        random_state=40)

    # fit model only once
    dtc_model.fit(X_train, y_train)
    pk.dump(dtc_model,open('static/model.pk','wb'))

@app.route('/prediksi', methods=['POST', 'GET'])
def submit_new_jurusan():
    # startup()
    akurasi,sensitivity,specifity=model()
    hasil_akurasi = akurasi
    hasil_sensitivity = sensitivity
    hasil_specifity = specifity
    model_results = ''
    if request.method == 'POST':
        selected_jurusan = request.form['selected_jurusan']
        selected_ipk = request.form['selected_ipk']
        selected_toefl = request.form['selected_toefl']
        selected_jeniskelamin = request.form['selected_jeniskelamin']
        selected_asal = request.form['selected_asal']

        # jurusan
        jurusan = 0
        if (selected_jurusan == 'Kimia Murni'):
            jurusan = 0
        if (selected_jurusan == 'Matematika'):
            jurusan = 1
        if (selected_jurusan == 'Fisika'):
            jurusan = 2
        if (selected_jurusan == 'Statistika'):
            jurusan = 3
        if (selected_jurusan == 'Ilmu Komputer'):
            jurusan = 4
        if (selected_jurusan == 'Biologi'):
            jurusan = 5

        # jenis kelamin
        jeniskelamin = 0
        if (selected_jeniskelamin == 'Laki - laki'):
            jeniskelamin = 0
        if (selected_jeniskelamin == 'Perempuan'):
            jeniskelamin = 1

        # asal
        asal = 0
        if (selected_asal == 'Luar Semarang'):
            asal = 0
        if (selected_asal == 'Semarang'):
            asal = 1

       # ipk
        ipk = 0
        if (selected_ipk == '<= 2.5'):
            ipk = 0.0
        if (selected_ipk == '> 2.5 dan <= 3.0'):
            ipk = 1.0
        if (selected_ipk == '> 3.0 dan <= 3.5'):
            ipk = 2.0
        if (selected_ipk == '> 3.5 dan <= 4.0'):
            ipk = 3.0

        # ipk
        toefl = 0
        if (selected_toefl == '<= 420'):
            toefl = 0.0
        if (selected_toefl == '> 420 dan <= 480'):
            toefl = 1.0
        if (selected_toefl == '> 480 dan <= 520'):
            toefl = 2.0
        if (selected_toefl == '> 520'):
            toefl = 3.0

        # build new array
        data_wisuda = [[jurusan, ipk, toefl, jeniskelamin, asal]]

        # add predict
        dtc_model = pk.load(open('static/model.pk','rb'))
        Y_pred = dtc_model.predict(data_wisuda)

        if (Y_pred == 1):
            model_results = "Tidak Lulus Tepat Waktu (> 4)"
        elif (Y_pred == 0):
            model_results = "Lulus Tepat Waktu (<= 4)"
        else:
            model_results = "Kelas tidak diketahui"
        print(Y_pred)

        print(jurusan,jeniskelamin, asal, ipk ,toefl)
        return render_template('prediksi.html', model_results=model_results,
                               selected_jurusan=selected_jurusan,
                               selected_ipk=selected_ipk,
                               selected_toefl=selected_toefl,
                               selected_jeniskelamin=selected_jeniskelamin,
                               selected_asal=selected_asal, akurasi_hasil=Markup(hasil_akurasi), specifity_hasil=Markup(hasil_specifity), sensitivity_hasil=Markup(hasil_sensitivity))

    else:
        # set default passenger settings
        return render_template('prediksi.html', model_results='',
                               selected_jurusan=DEFAULT_JURUSAN,
                               selected_ipk=DEFAULT_IPK,
                               selected_toefl=DEFAULT_TOEFL,
                               selected_jeniskelamin=DEFAULT_JENKEL,
                               selected_asal=DEFAULT_ASAL,akurasi_hasil=Markup(hasil_akurasi), specifity_hasil=Markup(hasil_specifity), sensitivity_hasil=Markup(hasil_sensitivity))

def initree():
    tree = DecisionTree()
    tr_data, clss, attrs = tree.read_data('static/datadiskrit.csv')

    tree1 = tree.create_tree(tr_data, clss, attrs)

    tree.showTree(tree1, ' ')
    # tree.list_tree

    # In[7]:

    # a = tree
    tree_string = tree.list_tree

    tree_string

    # In[8]:

    # temp = {}
    # for aa in tree_string:
    #     print(aa, aa.count('|'))
    #     temp

    # In[9]:

    tree1

    # In[27]:

    # a = tree
    tree_dict = tree1

    tree_dict

    json.dump(tree.convertTree(tree_dict, None), open("static/converted_tree.json", "w+"))
#
#
# if __name__ == '__main__':
#     app.run()
