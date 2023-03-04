import gradio as gr
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# %matplotlib inline
import io

def importdata():
		#balance_data = pd.read_csv(io.BytesIO(uploaded['heart_disease_data.csv']))
			balance_data = pd.read_csv('heart_disease_data.csv')
		# Printing the dataswet shape
			print ("Dataset Length: ", len(balance_data))
			print ("Dataset Shape: ", balance_data.shape)
		
		# Printing the dataset obseravtions
			print ("Dataset: ",balance_data.head())
		
			return balance_data
def splitdatasetL(heart_data, input_data):
  X = heart_data.drop(columns='target', axis=1)
  Y = heart_data['target']
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
  model = LogisticRegression()
  model.fit(X_train, Y_train)

  input_data_as_numpy_array= np.asarray(input_data)

  # reshape the numpy array as we are predicting for only on instance
  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

  prediction = model.predict(input_data_reshaped)

  return prediction[0]
def splitdataset(balance_data):

	# Separating the target variable
	X = balance_data.values[:, 0:13]
	Y = balance_data.values[:, 13]

	# Splitting the dataset into train and test
	X_train, X_test, y_train, y_test = train_test_split(
	X, Y, test_size = 0.3, random_state = 100)
	
	return X, Y, X_train, X_test, y_train, y_test
def train_using_gini(X_train, X_test, y_train):

	clf_gini = DecisionTreeClassifier(criterion = "gini",
			random_state = 100,max_depth=3, min_samples_leaf=5)

	clf_gini.fit(X_train, y_train)
	return clf_gini

def tarin_using_entropy(X_train, X_test, y_train):

	clf_entropy = DecisionTreeClassifier(
			criterion = "entropy", random_state = 100,
			max_depth = 3, min_samples_leaf = 5)

	clf_entropy.fit(X_train, y_train)
	return clf_entropy


# Function to make predictions
def prediction(X_test, clf_object):

	# Predicton on test with giniIndex
	y_pred = clf_object.predict(X_test)
	print("Predicted values:")
	print(y_pred)
	return y_pred
def RandomF(X_train, y_train, X_test):
  rf_clf = RandomForestClassifier(n_estimators=1000, random_state=42)
  rf_clf.fit(X_train, y_train)
  pred = rf_clf.predict(X_test)
  return pred
def SBM(df, X_test):
  X = df.drop('target', axis=1)
  y = df['target']
  X_train, X_T, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.fit_transform(X_test)
  svm = SVC(kernel='rbf', gamma=0.1)
  svm.fit(X_train_scaled, y_train)
  y_pred = svm.predict(X_test_scaled)
  return y_pred

def SBF(new_data):
  df = pd.read_csv('heart_disease_data.csv')
  X = df.drop('target', axis=1)
  y = df['target']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
  svm = SVC(kernel='linear')
  svm.fit(X_train, y_train)
  y_pred = svm.predict(new_data)
  print(y_pred)
  print(y_pred[0])
  print("MEasdasdaGASDASD")
  return y_pred[0]

def heart(age, gender, chestpaintype, restingbloodpressure, serumcholestrol, fastingbloodsugar, resting_ecg_result, maximumheartrate, exerciseinduced_angina, oldpeak, slope, ca, thal):
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)
    
    fbs = 1 if fastingbloodsugar > 120 else 0
    g   = 0 if gender == "Female" else 1
    exang   = 0 if exerciseinduced_angina == "No" else 1
    cp = 0
    if chestpaintype == "Typical Angina" :
      cp = 0
    elif chestpaintype == "Non Typical Angina" :
      cp = 1
    elif chestpaintype == "Non Anginal Pain" :
      cp = 2
    else :
      cp = 3
    ecg = 0
    if resting_ecg_result == "0 - Nothing to note" :
      ecg = 0
    elif resting_ecg_result == "1 - ST-T abnormality" :
      ecg = 1
    else :
      ecg = 2  
    
    XX = np.array([age, g, cp, restingbloodpressure, serumcholestrol, fbs, ecg, maximumheartrate, exang, oldpeak, slope, ca, thal])
    X_test[1][0] = age
    X_test[1][1] = g
    X_test[1][2] = cp
    X_test[1][3] = restingbloodpressure
    X_test[1][4] = serumcholestrol
    X_test[1][5] = fbs
    X_test[1][6] = ecg
    X_test[1][7] = maximumheartrate
    X_test[1][8] = exang
    X_test[1][9] = oldpeak
    X_test[1][10] = slope
    X_test[1][11] = ca
    X_test[1][12] = thal
    new_data = pd.DataFrame({'age':[age],'sex':[g],'cp':[cp],'trestbps':[restingbloodpressure],
                         'chol': [serumcholestrol],'fbs':[fbs],'restecg': [ecg],
                         'thalach':[maximumheartrate],'exang':[exang],'oldpeak': [oldpeak],
                         'slope':[slope],	'ca':[ca],	'thal':[thal]})
    y_pred_gini = prediction(X_test, clf_gini)
    k = RandomF(X_train, y_train, X_test)
    #m = SBM(data, new_data)
    m = SBF(new_data)
    print("ASDASDASDADS")
    print(type(m))
    #m = 0
    pred = splitdatasetL(data, XX)
    if y_pred_gini[1] == 1.0:
      SD = "Based on our Decision Tree Machine Learning model which has an accuracy of 82.42%, you have high chances of having heart disease"
    else:
      SD = "Based on our Decision Tree Machine Learning model which has an accuracy of 82.42%, you are less likely to have heart disease"
    if pred == 1:
      SL = "Based on our Logistic Regression Machine Learning model which has an accuracy of 81.97%, you have high chances of having heart disease"
    else:
      SL = "Based on our Logistic Regression Machine Learning model which has an accuracy of 81.97%, you are less likely to have heart disease"
    if k[1] == 1:
      SR = "Based on our Random Forest Machine Learning model which has an accuracy of 82.42%, you have high chances of having heart disease"
    else:
      SR = "Based on our Random Forest Machine Learning model which has an accuracy of 82.42%, you are less likely to have heart disease"
    if m == 1:
      SS = "Based on our SVM Machine Learning model which has an accuracy of 89.01%, you have high chances of having heart disease"
    else:
      SS = "Based on our SVM Machine Learning model which has an accuracy of 89.01%, you are less likely to have heart disease"
    
    models = ['Logistic Regression','Decision Tree','SVM','Random Forest']
    accuracies = [81.97,82.42,89.01,82.42]

    fig, ax = plt.subplots(figsize = (40,40))
    ax.bar(models, accuracies)
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')
    ax.set_title('Machine LearninModels Accuracy')

    return SL, SD, SS, SR, fig

gr.Interface(
    fn=heart,
    inputs=["number", gr.Radio(["Male", "Female"]),gr.Dropdown(["Typical Angina", "Non Typical Angina", "Non Anginal Pain", "Asymptomatic"]), "number", "number", "number", gr.Dropdown(["0 - Nothing to note", "1 - ST-T abnormality", "2 - Possible or definite left ventricular hypertrophy"]), "number", gr.Radio(["No", "Yes"]), "number" , "number", "number", "number"],
    outputs=[gr.outputs.Label(label="Logistic Regression", type="text"),gr.outputs.Label(label="Decision Tree", type="auto"),gr.outputs.Label(label="Random Forest", type="text"),gr.outputs.Label(label="SVM", type="auto"),"plot"],
server_name="0.0.0.0"
).launch()

interface.launch()