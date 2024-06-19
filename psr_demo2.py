import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from random import shuffle, seed

sys.path.insert(0, '../../fair_classification/')
import utils as ut
import loss_funcs as lf

def get_one_hot_encoding(attr_vals):
    encoder = OneHotEncoder(sparse_output=False)
    encoded_vals = encoder.fit_transform(np.array(attr_vals).reshape(-1, 1))
    return encoded_vals, {val: i for i, val in enumerate(encoder.categories_[0])}

def load_bank_data(filepath='/content/drive/MyDrive/Github/fair-classification2/fair-classification/disparate_impact/adult_data_demo/bank-full.csv', 
                   load_data_size=None, random_state=42, train_frac=0.7):
    bank = pd.read_csv(filepath)
    print("Initial data loaded")

    bank.loc[bank['marital'] != 'married', 'marital'] = 0
    bank.loc[bank['marital'] == 'married', 'marital'] = 1
    print("Marital status converted")

    bank['y'] = bank['y'].map({"no": 0, "yes": 1})
    print("Target variable converted")

    attrs = bank.columns
    int_attrs = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    sensitive_attrs = ['marital']
    attrs_to_ignore = ['marital', 'y']
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)

    X = []
    y = []
    x_control = {}

    attrs_to_vals = {}
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []
        elif k in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[k] = []

    y = bank.values[:, -1]

    for i in range(len(bank)):
        line = bank.iloc[i].values
        for j in range(0, len(line)):
            attr_name = attrs[j]
            attr_val = line[j]
            if attr_name in sensitive_attrs:
                x_control[attr_name].append(attr_val)
            elif attr_name in attrs_to_ignore:
                pass
            else:
                attrs_to_vals[attr_name].append(attr_val)
    print("Attributes and control variables separated")

    def convert_attrs_to_ints(d):
        for attr_name, attr_vals in d.items():
            if attr_name in int_attrs:
                continue
            attr_vals = [str(val) for val in attr_vals]
            uniq_vals = sorted(list(set(attr_vals)))
            val_dict = {val: i for i, val in enumerate(uniq_vals)}
            d[attr_name] = [val_dict[val] for val in attr_vals]

    convert_attrs_to_ints(x_control)
    convert_attrs_to_ints(attrs_to_vals)
    print("Attributes converted to integers")

    for attr_name in attrs_for_classification:
        attr_vals = attrs_to_vals[attr_name]
        if attr_name in int_attrs or attr_name == "native_country":
            X.append(attr_vals)
        else:
            attr_vals, _ = get_one_hot_encoding(attr_vals)
            for inner_col in attr_vals.T:
                X.append(inner_col)
    print("One-hot encoding applied")

    X = np.array(X, dtype=float).T
    y = np.array(y, dtype=float)
    for k, v in x_control.items():
        x_control[k] = np.array(v, dtype=float)

    perm = list(range(0, len(y)))
    seed(random_state)
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]
    print("Data shuffled")

    if load_data_size is not None:
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in x_control.keys():
            x_control[k] = x_control[k][:load_data_size]

    Z = np.expand_dims(x_control['marital'], axis=-1)

    Xtr, Xte, ytr, yte, Ztr, Zte = train_test_split(X, y, Z, train_size=train_frac, random_state=random_state)
    print("Train-test split applied")

    mm = MinMaxScaler()
    Xtr = mm.fit_transform(Xtr)
    Xte = mm.transform(Xte)
    print("Feature scaling applied")

    return Xtr, Xte, ytr, yte, Ztr.flatten(), Zte.flatten()

def test_bank_data():
    filepath = '/content/drive/MyDrive/Github/fair-classification2/fair-classification/disparate_impact/adult_data_demo/bank-full.csv'
    Xtr, Xte, ytr, yte, Ztr, Zte = load_bank_data(filepath)
    x_control = {'marital': Ztr}

    sm = SMOTE(random_state=42)
    Xtr, ytr = sm.fit_resample(Xtr, ytr)
    print("SMOTE applied to training data")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    clf.fit(Xtr, ytr)

    best_model = clf.best_estimator_
    print(f"Best model parameters: {clf.best_params_}")

    y_pred = best_model.predict(Xte)
    accuracy = accuracy_score(yte, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(yte, y_pred))

    return

def main():
    test_bank_data()

if __name__ == '__main__':
    main()
