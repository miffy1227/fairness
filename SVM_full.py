from psr_prepare_data import load_bank_data
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import os
import pickle

# 데이터 파일의 경로를 지정합니다.
filepath = '/content/fair-classification/disparate_impact/adult_data_demo/bank-full.csv'

# load_bank_data 함수를 호출할 때 데이터 파일의 경로를 전달합니다.
X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_bank_data(filepath, svm=True, random_state=42)

clf = svm.SVC(kernel='rbf', gamma=0.1)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_te)

print('pred : ')
print(pred)
print('clf.score : ')
print(clf.score(X_te, y_te))

svs = clf.support_vectors_
print(svs)
alphas = clf.dual_coef_
kern = clf.kernel

def save_svm(clf, save_dir='', filename='svm_model'):
    res = {}
    res['sv'] = clf.support_vectors_
    res['alpha'] = clf.dual_coef_
    res['kernel'] = clf.kernel
    res['gamma'] = clf._gamma
    res['coef0'] = clf.coef0
    res['degree'] = clf.degree
    res['intercept'] = clf.intercept_
    save_path = os.path.join(save_dir, filename+'.sm')
    with open(save_path, 'wb') as f:
        pickle.dump(res, f)

def load_svm(save_path):
    with open(save_path, 'rb') as f:
        aa = pickle.load(f)
    return aa

save_svm(clf)
sm = load_svm('svm_model.sm')
print('sm : ')
print(sm)

from kernel_fns import calculate_kernel
calculate_kernel(X_tr[:100])

pred_ = clf.predict(X_te)
def predict_svm(X_te, Xxv, alpha, intercept, kernel, **kwds):
    Ktest = calculate_kernel(Xxv, X_te, kernel, **kwds)
    dec_eval = Ktest.T.dot(alpha.flatten()) + intercept
    return np.sign(dec_eval).flatten()

pred = predict_svm(X_te, sm['sv'], sm['alpha'], sm['intercept'], sm['kernel'])
print('pred : ')
print(pred)
print('sum(pred == pred_)')
print(sum(pred == pred_))
