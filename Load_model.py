from audioop import avg
import cmath
from fileinput import filename
from gc import enable
import sys
from matplotlib.pyplot import axis
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, auc 
import warnings
import pickle
warnings.filterwarnings("ignore")
from fastai.tabular.all import *

dataPath = ('/Volumes/D/NCS_2021-2023/DGID/dataset/CIC_2019/dataset/')
modelPath = '/Volumes/D/NCS_2021-2023/DGID/10_2/'
fileName = 'train.csv'
df_train = pd.read_csv('/Volumes/D/NCS_2021-2023/DGID/dataset/CIC_2019/dataset/train.csv')
df_test = pd.read_csv('/Volumes/D/NCS_2021-2023/safecomp/CIC-2018-best/test.csv')

labels = ['SQL Injection', 'Infilteration', 'DoS attacks-SlowHTTPTest','DoS attacks-GoldenEye', 'Bot', 'DoS attacks-Slowloris','Brute Force -Web', 'DDOS attack-LOIC-UDP', 'Benign','Brute Force -XSS']

cat_names = ['Dst Port', 'Protocol']
y_names = 'Label'
cont_names = ['Flow Duration', 'Tot Fwd Pkts',
              'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
              'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
              'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
              'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
              'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
              'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
              'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
              'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
              'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
              'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
              'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
              'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
              'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
              'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
              'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
              'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
              'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
              'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
              'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
              'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']

procs = [Categorify, FillMissing, Normalize]
y_block = CategoryBlock()

print("Loading model")
gmodel = load_learner('/Volumes/D/NCS_2021-2023/safecomp/models/DLID1')
data = df_test
data[y_names] = data[y_names].astype('category')
print("Loading Testing dataset: \n",data[y_names].value_counts(), "\n")
yLabels = data['Label']
print("Preparing test_dl...")
dl = gmodel.dls.test_dl(data, with_labels=True, drop_last=False)
print("Predicting...")
start = time.time()
nn_preds, tests, clas_idx = gmodel.get_preds(dl=dl, with_loss=False, with_decoded=True)
loss, acc, precision, f1, recall, roc = gmodel.validate(dl=dl)
elapsed = time.time() - start
print('Accuracy: {:.2f}%; precision: {:.2f}%; F1: {:.2f}%; Recall: {:.2f}%; roc-auc: {:.2f}%; elapsed: {:.2f} s'.format(acc*100, precision*100, f1*100, recall*100, roc*100,  elapsed ))
interp = ClassificationInterpretation.from_learner(gmodel,dl=dl)
print("Confusion Matrix:\n", interp.confusion_matrix())


