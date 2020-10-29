# Importing liberaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Datasets
dataset = pd.read_csv('Python/Data.csv')

# Seperating dependent and independent vairiable
# Independent vairiable
x = dataset.iloc[:,:-1].values

#Dependent vairible
y = dataset.iloc[:,3].values


# Dealing with the missing data by replacing the missing data by the mean of that corresponding column

# we will use SimpleImputer from scikit-learn liberary.
from sklearn.impute import SimpleImputer 
# Instanciating the imposter class
imputer = SimpleImputer(missing_values = np.nan,strategy = 'mean')
# selecting the columns with the missing data
imputer.fit(x[:,1:3])
# Replacing the missind data withe the mean of the corresponding column's mean
x[:,1:3] = imputer.transform(x[:,1:3])