# AI-Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Sample data: [height, weight, shoe_size]
features = [
    [181, 80, 44], [177, 70, 43], [160, 60, 38],
    [154, 54, 37], [166, 65, 40], [190, 90, 47],
    [175, 64, 39], [177, 70, 40], [159, 55, 37],
    [171, 75, 42], [181, 85, 43]
]

labels = [
    'male', 'male', 'female', 'female', 'male',
    'male', 'female', 'female', 'female', 'male', 'male'
]

# Initialize models
model_dt = DecisionTreeClassifier()
model_svm = SVC()
model_pc = Perceptron()
model_knn = KNeighborsClassifier()

# Train each model
model_dt.fit(features, labels)
model_svm.fit(features, labels)
model_pc.fit(features, labels)
model_knn.fit(features, labels)

# Make predictions
pred_dt = model_dt.predict(features)
pred_svm = model_svm.predict(features)
pred_pc = model_pc.predict(features)
pred_knn = model_knn.predict(features)

# Calculate accuracy
score_dt = accuracy_score(labels, pred_dt) * 100
score_svm = accuracy_score(labels, pred_svm) * 100
score_pc = accuracy_score(labels, pred_pc) * 100
score_knn = accuracy_score(labels, pred_knn) * 100

# Display results
print(f"Decision Tree Accuracy: {score_dt:.2f}%")
print(f"SVM Accuracy: {score_svm:.2f}%")
print(f"Perceptron Accuracy: {score_pc:.2f}%")
print(f"KNN Accuracy: {score_knn:.2f}%")

# Find best model (excluding DecisionTree)
accuracies = [score_svm, score_pc, score_knn]
names = ['SVM', 'Perceptron', 'KNN']
best_model = names[np.argmax(accuracies)]

print(f"Best performing model: {best_model}")

# Predicting a new sample
new_sample = np.array([[175, 65, 41]])  #male sample
print("Predictions sample: [175, 65, 41]")
print(model_dt.predict(new_sample)[0])
