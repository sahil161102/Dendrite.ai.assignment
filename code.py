
import pandas as pd
import numpy as np
import json 


#LOADING JSON

# file = open("C:/Users/asus/Downloads/Internshala_DS_test/Internshala/algoparams_from_ui.json")
# json_file = file.read()
# algo_params = json.loads(json_file)
# file.close()
# print(algo_params)
#algo_params = pd.read_json("C:/Users/asus/Downloads/Internshala_DS_test/Internshala/algoparams_from_ui.json")
with open("C:/Users/asus/Downloads/Internshala_DS_test/Internshala/algoparams_from_ui.json") as data:
    algo_params = json.load(data)
# print(algo_params)



# EXTRACTING TARGET, TYPE AND PREDICTION_TYPE FROM THE JSON
target = ""
type = ""
prediction_type = ""

target = algo_params["design_state_data"]["target"]["target"]
type = algo_params["design_state_data"]["target"]["type"]
prediction_type = algo_params["design_state_data"]["target"]["prediction_type"]

#READING DATASET
dataset = pd.read_csv("C:/Users/asus/Downloads/Internshala_DS_test/Internshala/iris.csv")
# print(dataset)
features = dataset.columns
# print(features)

tarIndex = -1
len = features.size
for i in range(0,len):
    if(features[i] == target):
        tarIndex = i
# print(tarIndex)

# impute_method = []
# for i in range(0,len):
#     for feature_name in algo_params["design_state_data"]["feature_handling"]["feature_name"]:
#         if(features[i] == feature_name):
#             impute_method.append(algo_params["design_state_data"]["feature_handling"]["feature_details"]["impute_with"])

# print(impute_method)

# IMPUTING MISSING VALUES
# from sklearn.impute import SimpleImputer 
# imputer = SimpleImputer(missing_values=np.nan , strategy = "mean")
# imputer.fit(dataset[:,:-1])
# dataset[:,:-1] = imputer.transform(dataset[:,:-1])
# print(dataset)

# # ENCODING/TOKENIZING
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers =[('encoder', OneHotEncoder(), [0])] , remainder= 'passthrough')
# dataset = np.array(ct.fit_transform(dataset))
# print(dataset)

# SEPARATING INDEPENDENT AND DEPENDENT VARIABLES
x = dataset.iloc[0:,0:tarIndex].values
xp = dataset.iloc[0:,tarIndex+1:len+1].values
# print(xp,"\n")
x = np.append(x,xp, axis = 1)
y = dataset.iloc[:,tarIndex:tarIndex+1].values
# print(x,"\n",y)

# PARSING AND EXTRACTING ALGORITHMS NAME
algorithms = []
for i in algo_params["design_state_data"]["algorithms"]:
    algorithms.append(i)
# print(algorithms)

#REGRESSION ALGORITHMS
regAlgo = ['RandomForestRegressor', 'GBTRegressor', 'LinearRegression', 'RidgeRegression', 'LassoRegression', 'ElasticNetRegression', 'xg_boost', 'DecisionTreeRegressor', 'SGD', 'KNN', 'extra_random_trees', 'neural_network']

#CLASSIFICATION ALGORITHMS
classAlgo = ['RandomForestClassifier', 'GBTClassifier','LogisticRegression','xg_boost','DecisionTreeClassifier', 'SVM', 'SGD', 'KNN', 'extra_random_trees', 'neural_network']


