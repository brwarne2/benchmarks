from pandas import read_csv
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
import numpy as np
from sklearn.impute import KNNImputer


# load data
dataframe = np.genfromtxt('cnae9.csv', delimiter=',')
print(dataframe.shape)
# shuffle
np.random.seed(0)
np.random.shuffle(dataframe)
# fold
kf = KFold(n_splits=5)
# impute missingness
# imputer = KNNImputer()
# imputer.fit(dataframe)
# trainTrans = imputer.transform(dataframe)
# split and write
for i, (train_index, test_index) in enumerate(kf.split(dataframe)):
    training = dataframe[train_index]
    testing = dataframe[test_index]
    # output files
    file_name = 'cnae9_test_{}.csv'.format(i)
    # file_name = 'cnae9_train_{}.csv'.format(i)
    np.savetxt(file_name, testing, delimiter=',')
