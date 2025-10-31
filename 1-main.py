import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#### 1. read form the csv file
data = pd.read_csv("housing.csv")
# print(data)

#### 2. split the data: train and test
data['income_cat'] = pd.cut(data['median_income'], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1,2,3,4,5])
# print(data)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_indices, test_indices in split.split(data, data['income_cat']):
    train_data = data.iloc[train_indices].copy().drop("income_cat", axis=1)
    test_data = data.iloc[test_indices].copy().drop("income_cat", axis=1)

# print("training data:\n ",train_data)
# print("testing data: \n",test_data)

#### we will work with the copy of train_data:
housing = train_data.copy()
# print(housing.head())

#### 3. Seperate Features and Labels:
labels = housing["median_house_value"].copy()
features = housing.drop("median_house_value", axis=1)
# print("features \n", features)
# print("labels \n", labels)

#### 4. List numerical and Categorical columns
num_att = features.drop("ocean_proximity", axis=1).columns.tolist()
cat_att = ["ocean_proximity"]

#### 5. Make the pipelines

#### numerical attributes' pipeline
numpipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scaling", StandardScaler())
])

#### Categorical attributes' pipepline
catpipe = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

#### 6. combine the pipeline.
fullpipeline = ColumnTransformer([
    ("num", numpipe, num_att),
    ("cat", catpipe, cat_att)
])

#### 7. fit_transform the fullpipeline
housing_prepared = fullpipeline.fit_transform(features)
# print("success!")
# print(housing_prepared)


#### Get the features (col_names):
feature_names = fullpipeline.get_feature_names_out()
# print(feature_names)
feature_names = [x.split("__")[-1] for x in feature_names]
# print(feature_names)

# create the dataframe:
features_furnished = pd.DataFrame(housing_prepared, columns=feature_names, index = features.index)


