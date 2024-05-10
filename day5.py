"""   Created on Wed May  8 16:18:03 2024

# @author       :   Dr Hamed Aghapanah  , PhD bio-electrics

# @affiliation  :  Isfahan University of Medical Sciences

# """

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Number of samples
# num_samples = 1000
# # Number of features for each sample
# num_features = 2  # Reduced to 2 for visualization

# # Generate random 2D signals (features)
# X = np.random.rand(num_samples, num_features)

# # Generate random labels (binary classification)
# y = np.random.randint(2, size=num_samples)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a random forest classifier
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# # Evaluate model performance
# train_accuracy = accuracy_score(y_train, model.predict(X_train))
# test_accuracy = accuracy_score(y_test, model.predict(X_test))
# print("Training Accuracy:", train_accuracy)
# print("Testing Accuracy:", test_accuracy)

# # Visualization
# plt.figure(figsize=(12, 6))

# # Plot data points with true labels
# plt.subplot(1, 2, 1)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
# plt.title('True Labels')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.colorbar(label='Label')

# # Plot data points with predicted labels
# plt.subplot(1, 2, 2)
# plt.scatter(X[:, 0], X[:, 1], c=model.predict(X), cmap='coolwarm', edgecolors='k')
# plt.title('Predicted Labels (Model Output)')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.colorbar(label='Label')

# plt.tight_layout()
# plt.show()

# # Evaluate model performance
# train_accuracy = accuracy_score(y_train, model.predict(X_train))
# test_accuracy = accuracy_score(y_test, model.predict(X_test))
# print("Training Accuracy:", train_accuracy)
# print("Testing Accuracy:", test_accuracy)

# # Detect underfitting, overfitting, or fine-tuning
# if train_accuracy < 0.8:  # Adjust this threshold based on your requirements
#     print("Model may be underfitting.")
# elif train_accuracy > test_accuracy:
#     print("Model may be overfitting.")
# else:
#     print("Model is fine-tuned and can generalize well to unseen data.")




# # =============================================================================
# # 
# # =============================================================================
# plt.pause(2)
# plt.close('all')
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_squared_error

# # Generate synthetic data
# np.random.seed(0)
# X = np.linspace(0, 10, 100).reshape(-1, 1)
# y = np.sin(X) + np.random.normal(0, 0.1, size=X.shape)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Function to train and evaluate models of different polynomial degrees
# def train_and_evaluate_model(degree):
#     # Transform features to polynomial features
#     poly_features = PolynomialFeatures(degree=degree)
#     X_train_poly = poly_features.fit_transform(X_train)
#     X_test_poly = poly_features.transform(X_test)
    
#     # Train a linear regression model
#     model = LinearRegression()
#     model.fit(X_train_poly, y_train)
    
#     # Evaluate model performance
#     train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train_poly)))
#     test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test_poly)))
    
#     return model, train_rmse, test_rmse

# # Try models of different polynomial degrees
# degrees = range(1,50+1,2)
# models = []
# train_rmse_values = []
# test_rmse_values = []

# for degree in degrees:
#     model, train_rmse, test_rmse = train_and_evaluate_model(degree)
#     models.append(model)
#     train_rmse_values.append(train_rmse)
#     test_rmse_values.append(test_rmse)

# # Visualize results
# plt.figure(2)

# # Plot training and testing errors
# plt.subplot(1, 2, 1)
# plt.plot(degrees, train_rmse_values, label='Training RMSE', marker='o')
# plt.plot(degrees, test_rmse_values, label='Testing RMSE', marker='o')
# plt.title('Training and Testing Errors')
# plt.xlabel('Polynomial Degree')
# plt.ylabel('Root Mean Squared Error')
# plt.legend()

# # Plot models
# plt.subplot(1, 2, 2)
# plt.scatter(X_train, y_train, label='Training Data', color='blue')
# plt.scatter(X_test, y_test, label='Testing Data', color='red')
# x_values = np.linspace(0, 10, 1000).reshape(-1, 1)
# for degree, model in zip(degrees, models):
#     plt.figure(2)
#     plt.subplot(1, 2, 2)
#     poly_features = PolynomialFeatures(degree=degree)
#     x_poly = poly_features.fit_transform(x_values)
#     y_poly = model.predict(x_poly)
#     plt.plot(x_values, y_poly, label=f'Degree {degree}')
    
#     if degree==degrees[0]:
#         plt.figure(3);plt.subplot(1, 2, 1)
#         plt.plot(x_values, y_poly, label=f'Degree {degree}')
#         plt.scatter(X_train, y_train, label='Training Data', color='blue')
#         plt.scatter(X_test, y_test, label='Testing Data', color='red')
#         plt.title('Model Fits')
#         plt.xlabel('Feature')
#         plt.ylabel('Label')
#         plt.legend()
        
#     if degree==degrees[-1]:
#         plt.figure(3);plt.subplot(1, 2, 2)
#         plt.plot(x_values, y_poly, label=f'Degree {degree}')
#         plt.scatter(X_train, y_train, label='Training Data', color='blue')
#         plt.scatter(X_test, y_test, label='Testing Data', color='red')
        
#         plt.title('Model Fits')
#         plt.xlabel('Feature')
#         plt.ylabel('Label')
#         plt.legend()

# plt.tight_layout()
# plt.show()




# =============================================================================
# Regression
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

plt.close('all')
# ساخت یک مجموعه داده ساختگی برای مثال
np.random.seed(0)
n_samples = 10
X = np.random.rand(n_samples, 3)  # سه ویژگی تصادفی برای هر نمونه
coefficients = np.array([5, 3, 2])  # ضرایب واقعی برای هر ویژگی
intercept = 2  # عدد ثابت
noise = np.random.randn(n_samples)  # اضافه‌کردن نویز به داده‌ها
y = intercept + np.dot(X, coefficients) + noise  # مقدار وابسته به صورت خطی

# تقسیم داده‌ها به داده‌های آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ساخت مدل رگرسیونی خطی و آموزش آن
model = LinearRegression()
model.fit(X_train, y_train)

# پیش‌بینی برای داده‌های آموزش و آزمون
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# محاسبه معیارها
r_squared = r2_score(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
pearson_corr, _ = pearsonr(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)

# نمایش داده‌های واقعی و پیش‌بینی شده
plt.figure(5)

# داده‌های آموزش
plt.scatter(y_train, y_pred_train, color='blue', alpha=0.5, label='Train Data')
# داده‌های آزمون
plt.scatter(y_test, y_pred_test, color='red', alpha=0.5, label='Test Data')
# خط مرجع برای مقایسه
plt.plot([min(y), max(y)], [min(y), max(y)], color='black', linestyle='--')

# Plot the line y = ax + b
a = model.coef_[0]  # Slope of the line
b = model.intercept_  # Y-intercept

plt.legend()
plt.grid(True)
plt.show()

print("R-squared:", r_squared)
print("MSE:", mse)
print("Pearson correlation coefficient:", pearson_corr)
print("MAE:", mae)


# Draw lines from test data points to the regression line
for i in range(len(y_test)):
    plt.plot([y_test[i], y_pred_test[i]], [y_test[i], y_pred_test[i]], color='gray', linestyle='--', linewidth=0.5)
# =============================================================================
# exaple 2 
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.figure(4)

# Generate sample data
np.random.seed(42)
n_samples = 100
X_A = np.random.normal(0, 1, (n_samples // 2, 2))
X_B = np.random.normal(2, 1, (n_samples // 2, 2))
X = np.concatenate([X_A, X_B], axis=0)
y = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)], axis=0)

# Create the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the model parameters
a = model.coef_[0]  # Slope of the line
b = model.intercept_  # Intercept of the line

# Plot the data and the regression line
# plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Class A')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class B')

# Plot the regression line
x1 = np.min(X[:, 0])
x2 = np.max(X[:, 0])
y1 = a * x1 + b
y2 = a * x2 + b
plt.plot([x1, x2], [y1, y2], color='green', label='Regression Line')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linear Regression with Two Classes')
plt.legend()
plt.grid(True)
plt.show()


# =============================================================================
# classification
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the evaluation metrics
print("دقت (Accuracy):", accuracy)
print("میانگین دقت (Mean Accuracy):", np.mean(accuracy))
print("میانگین F1-Score:", f1)
print("ماتریس سردرگمی (Confusion Matrix):\n", conf_matrix)

# Visualize the dataset and classification results
plt.figure(7)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title("Iris Dataset with Decision Tree Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# =============================================================================
# example 2 classfication
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the accuracy and F1-score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

# Calculate Shannon Entropy
unique_labels, counts = np.unique(y_test, return_counts=True)
probabilities = counts / len(y_test)
shannon_entropy = -np.sum(probabilities * np.log2(probabilities))

# Print the results
print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy)
print("F1-Score:", f1)
print("Shannon Entropy:", shannon_entropy)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(np.arange(3), iris.target_names)
plt.yticks(np.arange(3), iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# =============================================================================
#  detection
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

plt.close('all')
n_sampless = [15,20, 200,700,1000,1500,2000,5000,10000]
for iii in range(len (n_sampless)):
    n_samples1= n_sampless[iii]
    # تولید داده‌های مصنوعی دو کلاسه
    X, y = make_classification(n_samples=n_samples1, n_features=10, n_informative=5, 
                               n_redundant=2, n_classes=2, random_state=42)
    
    # تقسیم داده‌ها به آموزشی و آزمایشی
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55)
    
    # ایجاد و آموزش مدل درخت تصمیم
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    # پیش‌بینی برای داده‌های آزمایشی
    y_pred = clf.predict(X_test)
    
    # محاسبه ماتریس اغتشاش
    cm = confusion_matrix(y_test, y_pred)
    
    # محاسبه معیارهای ارزیابی برای هر کلاس
    tn, fp, fn, tp = cm.ravel()
    class0_precision = tp / (tp + fp)
    class0_recall = tp / (tp + fn)
    class0_f1 = 2 * (class0_precision * class0_recall) / (class0_precision + class0_recall)
    class0_accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    class1_precision = tn / (tn + fn)
    class1_recall = tn / (tn + fp)
    class1_f1 = 2 * (class1_precision * class1_recall) / (class1_precision + class1_recall)
    class1_accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # محاسبه ROC-AUC
    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # چاپ نتایج
    print("Confusion Matrix:")
    print(cm)
    
    print("Class 0 Metrics:")
    print("Precision:", class0_precision)
    print("Recall:", class0_recall)
    print("F1-Score:", class0_f1)
    print("Accuracy:", class0_accuracy)
    
    print("Class 1 Metrics:")
    print("Precision:", class1_precision)
    print("Recall:", class1_recall)
    print("F1-Score:", class1_f1)
    print("Accuracy:", class1_accuracy)
    
    print("ROC-AUC:", roc_auc)

            

    
 
    # رسم منحنی ROC
    plt.figure(200)
    plt.plot(fpr, tpr,   lw=2, label='sample = '+str(n_samples1) +' ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1],  lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontname="Times New Roman", fontsize=14)
    plt.legend(loc="lower right")
    plt.show()


    
    # رسم ماتریس اغتشاش
    plt.figure(10+iii)
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix", fontname="Times New Roman", fontsize=14)
    plt.colorbar()
    plt.xticks(np.arange(2), ["Class 0", "Class 1"])
    plt.yticks(np.arange(2), ["Class 0", "Class 1"])
    plt.xlabel("Predicted Label", fontname="Times New Roman", fontsize=14)
    plt.ylabel("True Label", fontname="Times New Roman", fontsize=14)
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red", fontname="Times New Roman", fontsize=22)
    
    plt.show()

 


# =============================================================================
#  detection 2 
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

plt.close('all')
n_featuress = [10,20,50,100,500,1000]
for iii in range(len (n_featuress)):
    n_features1= int(n_featuress[iii])
    # تولید داده‌های مصنوعی دو کلاسه
    X, y = make_classification(n_samples=1000, n_features=n_features1, n_informative=5, 
                               n_redundant=2, n_classes=2, random_state=42)
    
    # تقسیم داده‌ها به آموزشی و آزمایشی
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55)
    
    # ایجاد و آموزش مدل درخت تصمیم
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    # پیش‌بینی برای داده‌های آزمایشی
    y_pred = clf.predict(X_test)
    
    # محاسبه ماتریس اغتشاش
    cm = confusion_matrix(y_test, y_pred)
    
    # محاسبه معیارهای ارزیابی برای هر کلاس
    tn, fp, fn, tp = cm.ravel()
    class0_precision = tp / (tp + fp)
    class0_recall = tp / (tp + fn)
    class0_f1 = 2 * (class0_precision * class0_recall) / (class0_precision + class0_recall)
    class0_accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    class1_precision = tn / (tn + fn)
    class1_recall = tn / (tn + fp)
    class1_f1 = 2 * (class1_precision * class1_recall) / (class1_precision + class1_recall)
    class1_accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # محاسبه ROC-AUC
    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # چاپ نتایج
    print("Confusion Matrix:")
    print(cm)
    
    print("Class 0 Metrics:")
    print("Precision:", class0_precision)
    print("Recall:", class0_recall)
    print("F1-Score:", class0_f1)
    print("Accuracy:", class0_accuracy)
    
    print("Class 1 Metrics:")
    print("Precision:", class1_precision)
    print("Recall:", class1_recall)
    print("F1-Score:", class1_f1)
    print("Accuracy:", class1_accuracy)
    
    print("ROC-AUC:", roc_auc)

            

    
 
    # رسم منحنی ROC
    plt.figure(200)
    plt.plot(fpr, tpr,   lw=2, label='n_featuress = '+str(n_features1) +' ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1],  lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontname="Times New Roman", fontsize=14)
    plt.legend(loc="lower right")
    plt.show()


    
    # رسم ماتریس اغتشاش
    plt.figure(10+iii)
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix", fontname="Times New Roman", fontsize=14)
    plt.colorbar()
    plt.xticks(np.arange(2), ["Class 0", "Class 1"])
    plt.yticks(np.arange(2), ["Class 0", "Class 1"])
    plt.xlabel("Predicted Label", fontname="Times New Roman", fontsize=14)
    plt.ylabel("True Label", fontname="Times New Roman", fontsize=14)
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red", fontname="Times New Roman", fontsize=22)
    
    plt.show()


# =============================================================================
# decision tree 
# درختان تصمیم‌گیری: 

# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Generate sample data
X, y = make_blobs(n_samples=100, centers=3, n_features=5, random_state=42)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Plot the Decision Tree
plt.figure(1000)
plot_tree(clf, feature_names=['Size', 'Color'], class_names=['Small', 'Medium', 'Large'], filled=True)
plt.title('Decision Tree for Apple Classification', fontname="Times New Roman", fontsize=14)
plt.show()


# =============================================================================
# NN
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=42)
y = (y == 1).astype(int)  # Convert labels to 0 and 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the perceptron neural network
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.train_accuracy_history = []
        self.test_accuracy_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_pred = self.activation(linear_output)
                self.weights += self.learning_rate * (y[i] - y_pred) * X[i]
                self.bias += self.learning_rate * (y[i] - y_pred)

            # Evaluate the model on the training and testing sets
            train_accuracy = np.mean(self.predict(X_train) == y_train)
            test_accuracy = np.mean(self.predict(X_test) == y_test)
            self.train_accuracy_history.append(train_accuracy)
            self.test_accuracy_history.append(test_accuracy)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation(linear_output)
        return y_pred

    def activation(self, x):
        return np.where(x >= 0, 1, 0)

# Train the perceptron model
perceptron = Perceptron(learning_rate=0.01, epochs=100)
perceptron.fit(X_train, y_train)

# Evaluate the model
train_accuracy = np.mean(perceptron.predict(X_train) == y_train)
test_accuracy = np.mean(perceptron.predict(X_test) == y_test)
print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Visualize the decision boundary and accuracy trend
plt.figure(figsize=(12, 6))

# Plot the decision boundary
plt.subplot(1, 2, 1)
x1_min, x1_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
x2_min, x2_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                     np.linspace(x2_min, x2_max, 100))
Z = perceptron.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.4)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
plt.title('Perceptron Neural Network Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot the accuracy trend
plt.subplot(1, 2, 2)
plt.plot(perceptron.train_accuracy_history, label='Train Accuracy')
plt.plot(perceptron.test_accuracy_history, label='Test Accuracy')
plt.title('Accuracy Trend')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# =============================================================================
# NN 2 
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Use only the first two features
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

# Define the perceptron neural network
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.train_accuracy_history = []
        self.test_accuracy_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_pred = self.activation(linear_output)
                self.weights += self.learning_rate * (y[i] - y_pred) * X[i]
                self.bias += self.learning_rate * (y[i] - y_pred)

            # Evaluate the model on the training and testing sets
            train_accuracy = np.mean(self.predict(X_train) == y_train)
            test_accuracy = np.mean(self.predict(X_test) == y_test)
            self.train_accuracy_history.append(train_accuracy)
            self.test_accuracy_history.append(test_accuracy)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation(linear_output)
        return y_pred

    def activation(self, x):
        return np.where(x >= 0, 1, 0)

# Train the perceptron model
perceptron = Perceptron(learning_rate=0.01, epochs=100)
perceptron.fit(X_train, y_train)

# Evaluate the model
train_accuracy = np.mean(perceptron.predict(X_train) == y_train)
test_accuracy = np.mean(perceptron.predict(X_test) == y_test)
print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Visualize the decision boundary and accuracy trend
plt.figure(figsize=(12, 6))

# Plot the decision boundary
plt.subplot(1, 2, 1)
x1_min, x1_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
x2_min, x2_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                     np.linspace(x2_min, x2_max, 100))
Z = perceptron.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.4)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
plt.title('Perceptron Neural Network Decision Boundary')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

# Plot the accuracy trend
plt.subplot(1, 2, 2)
plt.plot(perceptron.train_accuracy_history, label='Train Accuracy')
plt.plot(perceptron.test_accuracy_history, label='Test Accuracy')
plt.title('Accuracy Trend')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# =============================================================================
#  NN3
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.metrics import f1_score

plt.close('all')

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the number of neurons in the hidden layers
n1 = 8;n2 = 4
# Create the neural network model
model = Sequential([
    Dense(X.shape[1], activation='relu', input_shape=(X.shape[1],)),
    Dense(n1, activation='relu'),
    Dense(n2, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes in the Iris dataset
])

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=0, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
f1 = f1_score(y_test, np.argmax(y_pred, axis=1), average='macro')
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}, Test F1-score: {f1:.4f}')

# Plot the training and validation accuracy
plt.figure(500)
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot the training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
 
plt.tight_layout()
plt.show()

file_name='w.h5'
model.save(file_name)

import os 
os.startfile(file_name)



# =============================================================================
# data spliting 
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
 
plt.close('all')

# Load the Iris dataset
iris = load_iris()
# Assuming you have your dataset loaded as 'X' and 'y'
# X = your_x_data
# y = your_y_data
X = iris.data[:, :2]
y = iris.target
# Initialize the 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Create a figure and axis
fig, ax = plt.subplots(10000)

# Iterate through the 10 folds
for i, (train_index, test_index) in enumerate(kf.split(X)):
   
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Plot the split
    ax.scatter(X_train[:, 0], X_train[:, 1], label=f'Fold {i+1} - Train', alpha=0.5,c='r')
    ax.scatter(X_test[:, 0], X_test[:, 1], label=f'Fold {i+1} - Test', alpha=0.5,c='g')
    plt.pause(2)
    
# Add labels and title
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('10-Fold Cross-Validation Split')
ax.legend()

# Show the plot
plt.show()

