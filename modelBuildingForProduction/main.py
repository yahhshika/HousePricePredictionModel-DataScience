import os
import pandas as pd
import numpy as np
import joblib
 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def getFullPipeLine(num, cat):
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scling", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("catsparx", OneHotEncoder())
    ])
    full_pipeline = ColumnTransformer([
        ("nums", num_pipeline, num),
        ("cats", cat_pipeline, cat)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    data = pd.read_csv("housing.csv")
    data["income_cat"] = pd.cut(data["median_income"], bins=[0.0,1.5,3.0,4.5,6.0, np.inf], labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_indices, test_indices in split.split(data, data["income_cat"]):
        train_data = data.iloc[train_indices].drop("income_cat", axis=1)
        test_data = data.iloc[test_indices].drop("income_cat", axis=1)
        test_data.to_csv("input.csv", index=False) # for the time being i am keeping this as input only but in real this should be for testing, input will come from real world

    housing = train_data.copy()

    # features and label extraction:
    features = housing.drop("median_house_value", axis = 1)
    labels = housing["median_house_value"].copy()

    # numerical attributes and categorical attribute extraction:
    num_att = features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_att = ["ocean_proximity"]

    # getting the pipeline: 
    fullpipeline = getFullPipeLine(num_att, cat_att)
    housing_transformed = fullpipeline.fit_transform(features)
    # print(housing_prepared)

    # getting the model ready
    ran_reg_model = RandomForestRegressor(random_state=42)
    ran_reg_model.fit(housing_transformed, labels)

    joblib.dump(ran_reg_model, MODEL_FILE)
    joblib.dump(fullpipeline, PIPELINE_FILE)

    print("Model trained and saved. re-run to get outputs")

else:
    input_data = pd.read_csv("input.csv")
    fullpipeline = joblib.load(PIPELINE_FILE)
    transformed_input = fullpipeline.transform(input_data)

    model = joblib.load(MODEL_FILE)
    predictions = model.predict(transformed_input)

    input_data["predictions"] = predictions

    input_data.to_csv("ouput.csv", index=False)
    print("Inference complete. Results saved to output.csv")
