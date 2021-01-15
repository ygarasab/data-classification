import h5py
import numpy as np
import pandas as pd

from algoritmos import codificador
from sklearn import model_selection

with h5py.File("../dados/dota2.h5", "r") as arquivo:
    X = pd.DataFrame(arquivo.get("X")[()])
    y = pd.DataFrame(arquivo.get("y")[()])

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

kf = codificador.KFoldStratifiedTargetEncoder()

# noinspection SpellCheckingInspection
# train, test = pd.read_csv("train.csv"), pd.read_csv("test.csv")

# X = train["Feature"].to_numpy().reshape(-1, 1)
# y = train["Target"].to_numpy()

results = kf.fit_transform(X_train.loc[:, 0:2].to_numpy().reshape(-1, 3), y_train.to_numpy().ravel())
