import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Laad de digits dataset
digits = datasets.load_digits()

# Split de data en de labels in trainings- en testsets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.33, random_state=42)

# Initialiseer de classifier
clf = svm.SVC(gamma=0.001, C=100)

# Train de classifier met de trainingsdata
clf.fit(X_train, y_train)

# Voorspel de labels voor zowel de trainings- als de testset
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

# Bereken en print de nauwkeurigheid op de trainingsdata
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"Nauwkeurigheid op de trainingsdata: {accuracy_train * 100:.2f}%")

# Bereken en print de nauwkeurigheid op de testdata
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Nauwkeurigheid op de testdata: {accuracy_test * 100:.2f}%")

# Toon een afbeelding uit de testset met de voorspelling
index_to_show = 4
predicted = clf.predict(X_test[index_to_show:index_to_show-1])
image = X_test[index_to_show].reshape(8, 8)
print(f"Voorspelde waarde van getoonde afbeelding: {predicted[0]}")
plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()