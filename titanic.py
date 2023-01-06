import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

train_path = "C:\\Users\\mashe\\Documents\\Titanic_datascience\\train.csv"
df = pd.read_csv(train_path)
test_path = "C:\\Users\\mashe\\Documents\\Titanic_datascience\\test.csv"
df1 = pd.read_csv(test_path)

print(df.describe())

#getting rid of useless data
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df1 = df1.drop(['Name', 'Ticket', 'Cabin'], axis=1)

#check for missing data
print(df.isnull().sum()) 
# 177 null in age, and 2 in embarked
print(df["Age"].mean())
#average is 29.7, so we will fill age null with 30

#find the most common of the 3 embarked, fill null with that
print(df[df == "S"].count())
print(df[df == "Q"].count())
print(df[df == "C"].count())
#S is the most common, so replace null with S

df = df.fillna(value={"Age":30, "Embarked":"S"})

#get information on trends within the data
print(df['Survived'].value_counts(normalize=True))
#38% survival rate

male = df[df["Sex"] == "male"]
female = df[df["Sex"] == "female"]
print(male['Survived'].value_counts(normalize=True))
print(female['Survived'].value_counts(normalize=True))
#men had a 19% chance of survival, women had a 74% chance of survival

ten = df[df["Age"] <= 10]["Survived"].sum()
twenty = df[df["Age"] <= 20]["Survived"].sum() - ten
thirty = df[df["Age"] <= 30]["Survived"].sum() - twenty - ten
forty = df[df["Age"] <= 40]["Survived"].sum() - thirty- twenty - ten
fifty = df[df["Age"] <= 50]["Survived"].sum() - forty - thirty- twenty - ten
sixty = df[df["Age"] <= 60]["Survived"].sum() - fifty - forty - thirty- twenty - ten
seventy = df[df["Age"] <= 70]["Survived"].sum() - sixty - fifty - forty - thirty- twenty - ten
eighty = df[df["Age"] <= 80]["Survived"].sum() - seventy - sixty - fifty - forty - thirty- twenty - ten

plt.xlabel("Age")
plt.ylabel("Survived")
plt.plot(['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80'], [ten, twenty, thirty, forty, fifty, sixty, seventy, eighty])
plt.show()
#we can see that majority of people who survived are in the range of 20-40

#before we can create the models, we must turn all data to numerical form
#this can be done with ordinal encoder
oe = OrdinalEncoder()
oe.fit(df[["Sex", "Embarked"]])
oe.fit(df1[["Sex", "Embarked"]])
df[["Sex", "Embarked"]] = oe.transform(df[["Sex", "Embarked"]])
df1[["Sex", "Embarked"]] = oe.transform(df1[["Sex", "Embarked"]])

#creating the model
X = df.drop(columns=['PassengerId', 'Survived']) 
Y = df["Survived"]
train_x, test_x, train_y, test_y = train_test_split(X, Y)

models = [RandomForestClassifier(), DecisionTreeClassifier(), 
KNeighborsClassifier(), SVC(), LogisticRegression()]

def pred(model, test_x, test_y, train_x, train_y):
    x = model.fit(train_x, train_y)
    pred_y = x.predict(test_x)
    score = accuracy_score(test_y, pred_y)
    #using accuracy score instead of mae, since mae is used for regression values, while accuracy score is used for classification
    return score

for i in models:
    print(pred(i, test_x, test_y, train_x, train_y))

#using this test method, we find that the most accurate model is random forest classifier

model = RandomForestClassifier()
model.fit(X, Y)
#fixing data for test case 
df1 = df1.fillna(value={"Age":30, "Fare":"36"})

predictions = model.predict(df1.drop(columns=['PassengerId']))
output = pd.DataFrame({'PassengerId':df1['PassengerId'], 'Survived':predictions})
output.to_csv('C:\\Users\\mashe\\Documents\\Titanic_datascience\\submission.csv', index=False)