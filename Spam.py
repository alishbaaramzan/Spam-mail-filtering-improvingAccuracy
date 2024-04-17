import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest classifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# read the data and replace null values with a null string
df1 = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/spamham.csv")
df = df1.where((pd.notnull(df1)), '')

# Categorize Spam as 0 and Not spam as 1
df['Category'] = df['Category'].map({'ham': 1, 'spam': 0})
# drop rows with missing values in 'Message' column
df = df.dropna(subset=['Message'])

# split data as label and text. System should be capable of predicting the label based on the text
df_x = df['Message']
df_y = df['Category']

# split the table - 80 percent for training and 20 percent for test size
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, train_size=0.8, test_size=0.2, random_state=4)

# feature extraction, coversion to lower case and removal of stop words using TFIDF VECTORIZER
tfvec = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_trainFeat = tfvec.fit_transform(x_train)
x_testFeat = tfvec.transform(x_test)

# SVM is used to model
classifierModel = LinearSVC()
classifierModel.fit(x_trainFeat, y_train)
predResult = classifierModel.predict(x_testFeat)

# GNB is used to model
classifierModel2 = MultinomialNB()
classifierModel2.fit(x_trainFeat, y_train)
predResult2 = classifierModel2.predict(x_testFeat)

# Random Forest Classifier
classifierModelRF = RandomForestClassifier(n_estimators=100, random_state=42)
classifierModelRF.fit(x_trainFeat, y_train)
predResultRF = classifierModelRF.predict(x_testFeat)

# Calc accuracy, converting to int - solves - cant handle mix of unknown and binary
actual_Y = y_test.values

print("~~~~~~~~~~SVM RESULTS~~~~~~~~~~")
# Accuracy score using SVM
print("Accuracy Score using SVM: {0:.4f}".format(accuracy_score(actual_Y, predResult) * 100))
# FScore MACRO using SVM
print("F Score using SVM: {0: .4f}".format(f1_score(actual_Y, predResult, average='macro') * 100))
cmSVM = confusion_matrix(actual_Y, predResult)
print("Confusion matrix using SVM:")
print(cmSVM)

print("~~~~~~~~~~MNB RESULTS~~~~~~~~~~")
# Accuracy score using MNB
print("Accuracy Score using MNB: {0:.4f}".format(accuracy_score(actual_Y, predResult2) * 100))
# FScore MACRO using MNB
print("F Score using MNB:{0: .4f}".format(f1_score(actual_Y, predResult2, average='macro') * 100))
cmMNb = confusion_matrix(actual_Y, predResult2)
print("Confusion matrix using MNB:")
print(cmMNb)

print("~~~~~~~~~~Random Forest RESULTS~~~~~~~~~~")
# Accuracy score using Random Forest
print("Accuracy Score using Random Forest: {0:.4f}".format(accuracy_score(actual_Y, predResultRF) * 100))
# FScore MACRO using Random Forest
print("F Score using Random Forest:{0: .4f}".format(f1_score(actual_Y, predResultRF, average='macro') * 100))
cmRF = confusion_matrix(actual_Y, predResultRF)
print("Confusion matrix using Random Forest:")
print(cmRF)
