from psr_prepare_data import *
import numpy as np
from sklearn import linear_model as lm
import matplotlib.pyplot as plt
import os
import pickle

# 데이터를 로드합니다.
X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_bank_data(filepath='/content/fair-classification/disparate_impact/adult_data_demo/bank-full.csv', svm=True, random_state=42, intercept=False)
print("Train data shape:", X_tr.shape)
print("Test data shape:", X_te.shape)

# 로지스틱 회귀 모델을 학습합니다.
clf = lm.LogisticRegression(max_iter=1000)  # max_iter 값을 1000으로 설정
clf.fit(X_tr, y_tr)

# 테스트 데이터를 예측합니다.
pred = clf.predict(X_te)
print("Predictions:", pred)

# 모델의 계수를 확인합니다.
coef = clf.coef_
print("Model Coefficients:", coef)

# 모델 저장 함수
def save_lr(clf, save_dir='', filename='lr_model'):
    res = {}
    res['coef'] = clf.coef_
    res['intercept'] = clf.intercept_
    save_path = os.path.join(save_dir, filename + '.sm')
    with open(save_path, 'wb') as f:
        pickle.dump(res, f)

# 모델 로드 함수
def load_lr(save_path):
    with open(save_path, 'rb') as f:
        aa = pickle.load(f)
    return aa

# 모델을 저장하고 로드합니다.
save_lr(clf)
sm = load_lr('lr_model.sm')
print("Loaded Model:", sm)

# 예측 함수
def predict_lr(X_te, coef, intercept):
    dec_eval = np.dot(X_te, coef.T) + intercept
    return np.sign(dec_eval).flatten()

# 로드한 모델로 예측합니다.
pred_loaded = predict_lr(X_te, sm['coef'], sm['intercept'])
print('sum(pred == pred_loaded):', sum(pred == pred_loaded))
print('Predictions using loaded model:', pred_loaded)
print('Original Predictions:', pred)
