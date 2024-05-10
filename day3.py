"""   Created on Wed May  1 10:46:01 2024

@author       :   Dr Hamed Aghapanah  , PhD bio-electrics

@affiliation  :  Isfahan University of Medical Sciences

"""

# =============================================================================
# NUMPY 
# =============================================================================

"""
NumPy Examples

This Python script contains examples demonstrating various applications of NumPy.

1. Array operations: Perform arithmetic and element-wise operations on arrays.
2. Mathematical functions: Use mathematical functions like sine, logarithm, and exponential.
3. Random number generation: Generate random numbers from different distributions.
4. Linear algebra operations: Perform matrix multiplication, inversion, and determinant calculation.
5. Data manipulation and processing: Manipulate and process data arrays.
6. Signal processing: Generate and process signals using NumPy functions.
7. Image processing: Create and display synthetic images using NumPy arrays.
8. Integration with other libraries: Demonstrate integration with pandas for data analysis.

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Array operations
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print("Array operations:")
print("Sum of arrays:", arr1 + arr2)
print("Element-wise multiplication:", arr1 * arr2)
print("Dot product:", np.dot(arr1, arr2))

# 2. Mathematical functions
print("\nMathematical functions:\n")
print("Sine of pi/2:", np.sin(np.pi/2))
print("Logarithm of 10:", np.log(10))
print("Exponential of 2:", np.exp(1))

# log (3.7)   in base 4 ==>
a= np.log (3.7)
b= np.log (4)
answer  = a/b

print(10*'=')

# 3. Random number generation
print("\nRandom number generation:")
print("Random numbers from uniform distribution:", np.random.rand(5))
print("Random numbers from normal distribution:", np.random.randn(5))

a=np.random.rand(3)
a=np.random.randn(3000000)
mean = np.mean(a)
min1 = np.min(a)
max1 = np.max(a)

print('Mean = ',mean)
print('min1 = ',min1)
print('max1 = ',max1)


# 4. Linear algebra operations
print("\nLinear algebra operations:")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A.shape)
print(np.shape (A))

print("A:",  A  )
print("B:",  B  )
print("Matrix multiplication:", np.matmul(A, B))
print("Matrix determinant:", np.linalg.det(A))
print("Matrix inverse:", np.linalg.inv(A))

# 5. Data manipulation and processing
print("\nData manipulation and processing:")
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Array shape:", data.shape)
print("Array mean:", np.mean(data))

print("Array mean along col:", np.mean(data , axis=0))
print("Array mean along rows:", np.mean(data, axis=1))


print("Array sum along col:", np.sum(data, axis=0))
print("Array sum along rows:", np.sum(data, axis=1))

# 6. Signal processing
print("\nSignal processing:")
t = np.linspace(0, 1, 1000)
# t= range (0,1000 )
# t=np.array(t)
# t=t/1000
signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
signal2 = np.sin(2 * np.pi * 15 * t)  # 15 Hz sine wave
fft_result = np.fft.fft(signal)  # Perform Fast Fourier Transform (FFT)
fft_result2 = np.fft.fft(signal2)  # Perform Fast Fourier Transform (FFT)
print("FFT result:", fft_result)
plt.subplot(1,3,1)
plt.plot(t)
plt.subplot(1,3,2)
plt.plot(signal)
plt.plot(signal2,'r')
plt.subplot(1,3,3)
plt.plot(fft_result)
plt.plot(fft_result2,'r')

plt.figure()
plt.plot(signal)
plt.plot(signal2,'r')

# 7. Image processing
print("\nImage processing:")
image = np.random.rand(100, 100)
plt.imshow(image, cmap='jet')
plt.axis('off')
plt.show()

# 8. Integration with other libraries
print("\nIntegration with other libraries:")
data = np.random.rand(5, 3)
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
print("DataFrame head:\n", df.head())
print("DataFrame statistics:\n", df.describe())
plt.imshow(data, cmap='gray')


# =============================================================================
# SciPy
# =============================================================================

"""
SciPy Examples

This Python script contains examples demonstrating various applications of SciPy.

1. Integration: Perform numerical integration using the quad function.
2. Optimization: Minimize a simple objective function using the minimize function.
3. Interpolation: Perform linear interpolation using the interp1d function.
4. Linear algebra: Solve a linear system of equations using the solve function.
5. Sparse matrices: Create a sparse matrix using the csr_matrix function.
6. Signal processing: Find peaks in a synthetic signal using the find_peaks function.
7. Statistical functions: Compute statistics from random data using functions from the stats module.
8. Image processing: Read and display an image using the face function.
9. Sparse graphs: Find connected components in a graph using the connected_components function.
10. Integration with other libraries: Demonstrate integration with pandas for data analysis.

"""

# from scipy.sparse 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.linalg import solve
from scipy.sparse import csr_matrix
from scipy.signal import find_peaks
from scipy.stats import norm
from scipy import misc
from scipy.sparse.csgraph import connected_components

# 1. Integration
def f(x):
    return np.sin(x)

result, error = quad(f, 0, np.pi)
print("Integration result:", result)

# 2. Optimization
def objective(x):
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

initial_guess = [0, 0]
result = minimize(objective, initial_guess)
print("Optimization result:", result.x)

# 3. Interpolation
x = np.linspace(0, 10, 10)
y = np.sin(x)
interpolated_function = interp1d(x, y, kind='linear')
print("Interpolated value at x=2.5:", interpolated_function(2.5))

plt.plot(x,y)
plt.plot(interpolated_function,'r')

# 4. Linear algebra
A = np.array([[2, 1], [1, 3]])
b = np.array([4, 5])
solution = solve(A, b)
print("Linear algebra solution:", solution)

# 5. Sparse matrices
data = np.array([1, 2, 3])
row_indices = np.array([0, 1, 2])
column_indices = np.array([0, 1, 2])
sparse_matrix = csr_matrix((data, (row_indices, column_indices)), shape=(3, 3))
print("Sparse matrix:")
print(sparse_matrix.toarray())

# 6. Signal processing
t = np.linspace(0, 10, 1000)
signal = 0.5* np.sin(2 * np.pi * 1 * t) + 0.5 * np.cos(2 * np.pi * 1.5 * t)
peaks, _ = find_peaks(signal)
print("Signal peaks:", peaks)
plt.plot(signal)
plt.plot(peaks , signal[peaks],'r*')


# 7. Statistical functions
data = np.random.normal(loc=0, scale=1, size=1000)
mean = np.mean(data)
std_dev = np.std(data)
cdf_at_zero = norm.cdf(0)
print("Mean:", mean)
print("Standard deviation:", std_dev)
print("CDF at zero:", cdf_at_zero)

# 8. Image processing
image = misc.face()
plt.imshow(image[:,:,0])
plt.axis('off')
plt.show()

# 9. Sparse graphs
adjacency_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
num_components, labels = connected_components(adjacency_matrix)
print("Number of connected components:", num_components)
print("Labels of nodes:", labels)

# 10. Integration with other libraries
data = np.random.rand(5, 3)
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
print("DataFrame head:\n", df.head())
print("DataFrame statistics:\n", df.describe())


# =============================================================================
# PANDAS 
# =============================================================================


"""
Pandas Examples

This Python script contains examples demonstrating various applications of pandas.

1. Data manipulation: Load a dataset, select specific columns, and filter rows based on conditions.
2. Data analysis: Compute descriptive statistics and perform groupby operations.
3. Data visualization: Create plots and visualizations using pandas and Matplotlib.
4. Data input/output (I/O): Read data from a CSV file and write data to an Excel file.
5. Time series analysis: Work with time series data, including date/time indexing and resampling.
6. Missing data handling: Handle missing values in a DataFrame using methods like dropna and fillna.
7. Data merging and joining: Merge multiple DataFrames using different join methods.
8. Data aggregation and transformation: Aggregate data using groupby operations and apply custom transformations.
9. Categorical data handling: Encode categorical variables and perform categorical data analysis.
10. Integration with other libraries: Use pandas in conjunction with other libraries like scikit-learn for data analysis and modeling.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Create sample data
data = {
    'column1': np.random.randint(1, 100, 10),
    'column2': np.random.rand(10),
    'column3': np.random.choice(['A', 'B', 'C'], 10),
    'column4': np.random.choice(['X', 'Y', 'Z'], 10),
    'category': np.random.choice(['Category1', 'Category2'], 10)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV file
df.to_csv('example.csv', index=False)
import os
os.startfile( 'example.csv')
print("Example CSV file 'example.csv' has been created.")


# import xlread

# 1. Data manipulation
# Load a dataset

df = pd.read_csv('example.csv')
df_hamed = pd.read_csv('hamed.csv')

# Select specific columns
selected_columns = df[['column1', 'column2']]

# Filter rows based on conditions
filtered_rows = df[df['column2'] > 0.5]

# 2. Data analysis
# Compute descriptive statistics
statistics = df.describe()

# Perform groupby operation
grouped_data = df.groupby('category').mean()

# 3. Data visualization
# Create a bar plot
df['column2'].value_counts().plot(kind='bar')
plt.title('Distribution of Column4')
plt.xlabel('Categories')
plt.ylabel('Count')
plt.show()

# 4. Data input/output (I/O)
# Read data from a CSV file
df = pd.read_csv('data.csv')

# Write data to an Excel file
df.to_excel('output.xlsx', index=False)
df.to_excel('output2.xlsx', index=True)

os.startfile('output.xlsx')
os.startfile('output2.xlsx')
# 5. Time series analysis
# Work with time series data
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
resampled_data = df.resample('M').sum()

# 6. Missing data handling
# Handle missing values
cleaned_data = df.dropna()
filled_data = df.fillna(0)

# 7. Data merging and joining
# Merge multiple DataFrames
df1=df
df2=df1
merged_data = pd.merge(df1, df2 )

# 8. Data aggregation and transformation
# Aggregate data using groupby
aggregated_data = df.groupby('category').sum()

# Apply custom transformations
df['new_column'] = df['column2'] * 2
df['new_column2'] = df['column2'] * df['column1']

# 9. Categorical data handling
# Encode categorical variables
encoded_data = pd.get_dummies(df['category'])

# Perform categorical data analysis
category_counts = df['category'].value_counts()

# 10. Integration with other libraries
# Use pandas with scikit-learn for data analysis
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = df[['feature1', 'feature2']]
y = df['target']
model.fit(X, y)





# =============================================================================
# Matplotlib
# =============================================================================


"""
Matplotlib Examples

This Python script contains examples demonstrating various applications of Matplotlib.

1. Line plots: Plot a simple line graph to visualize a relationship between variables.
2. Scatter plots: Create a scatter plot to show the distribution or relationship between two variables.
3. Bar plots: Generate a bar plot to compare values across different categories.
4. Histograms: Display the distribution of continuous data using bins and frequencies.
5. Pie charts: Represent proportions of different categories as slices of a pie.
6. Box plots: Visualize the distribution of data and identify outliers using quartiles and median.
7. Violin plots: Combine features of box plots and kernel density estimation to show the data distribution.
8. Heatmaps: Create a heatmap to visualize 2D data using color gradients.
9. Contour plots: Represent 3D data on a 2D plane using contours of equal values.
10. 3D plots: Generate a three-dimensional visualization of data for complex relationships or patterns.

"""

import matplotlib.pyplot as plt
import numpy as np

# 1. Line plots
plt.close('all')
plt.close(500)

fig = plt.figure(500)
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
plt.figure(2)
plt.plot(x, 0*y)
plt.figure(100)
y = np.cos(x)
plt.plot(x, y)

# 2. Scatter plots
fig = plt.figure(1000)
x = np.random.randn(100)
y = np.random.randn(100)
plt.scatter(x, y)
plt.title('Scatter Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

plt.pause(3)
plt.close('all')

# 3. Bar plots
plt.close('all')

fig = plt.figure(3)
categories = ['A', 'B', 'C', 'D']
values = [25, 30, 35, 40]
values2 = [75, 12, 40, 7]
plt.subplot( 211)

plt.bar(categories, values)
plt.ylim( [0,80])
plt.title(' نمودار بار چارت')

plt.subplot( 2,1,2)

plt.bar(categories, values2)
plt.title('Bar Plot')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.ylim( [0,80])
plt.show()

# 4. Histograms
fig = plt.figure()
data = np.random.randn(10000)
plt.hist(data, bins=30)
plt.title('Histogram')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# 5. Pie charts
plt.close('all')

fig = plt.figure()
sizes = [20, 30, 40, 10]
labels = ['A', 'B', 'C', 'D']
plt.pie(sizes, labels=labels, autopct='%1.3f%%')
plt.title('Pie Chart')
plt.show()

# 6. Box plots
plt.close('all')

fig = plt.figure()
data = np.random.randn(100)
plt.boxplot(data)
plt.title('Box Plot')
plt.ylabel('Values')
plt.show()

# 7. Violin plots
plt.close('all')

fig = plt.figure()
data = np.random.randn(100)
plt.violinplot(data)
plt.title('Violin Plot')
plt.ylabel('Values')
plt.show()

# 8. Heatmaps
fig = plt.figure()
data = np.random.rand(10, 10)
plt.imshow(data, cmap='viridis', interpolation='nearest')
plt.imshow(data, cmap='jet', interpolation='nearest')
plt.imshow(data, cmap='autumn', interpolation='nearest')
plt.colorbar()
plt.title('Heatmap')
plt.show()

# 9. Contour plots
plt.close('all')
fig = plt.figure()
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) + np.cos(Y)
plt.contour(X, Y, Z, cmap='viridis')
plt.colorbar()
plt.title('Contour Plot')
plt.show()

# 10. 3D plots
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)
ax.scatter(x, y, z)
ax.set_title('3D Scatter Plot')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()


# 11 Additional Application: Error bars
fig = plt.figure()

x = np.arange(10)
y = np.sin(x)
errors = np.random.rand(10) * 0.5  # Random error values

plt.errorbar(x, y, yerr=errors, fmt='o', capsize=5)
plt.title('Error Bars')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()




# 12 Additional Application: Subplots
fig = plt.figure()

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(8, 6))

# Plot sine function on the first subplot
axes[0].plot(x, y1, color='blue', label='Sine')
axes[0].set_title('Sine Function')
axes[0].set_xlabel('X-axis')
axes[0].set_ylabel('Y-axis')
axes[0].legend()

# Plot cosine function on the second subplot
axes[1].plot(x, y2, color='red', label='Cosine')
axes[1].set_title('Cosine Function')
axes[1].set_xlabel('X-axis')
axes[1].set_ylabel('Y-axis')
axes[1].legend()

# Adjust layout and display
plt.tight_layout()
plt.show()




# 13 Additional Application: Polar plots
fig = plt.figure()

theta = np.linspace(0, 2*np.pi, 100)
r = np.sin(3*theta) * np.cos(2*theta)

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, projection='polar')
ax.plot(theta, r)
ax.set_title('Polar Plot')
plt.show()


# 14 Additional Application: Stream plots
fig = plt.figure()

x = np.linspace(0, 2*np.pi, 100)
y = np.linspace(-2, 2, 10)
X, Y = np.meshgrid(x, y)
U = np.sin(X)
V = np.cos(Y)

plt.streamplot(X, Y, U, V, density=1.5)
plt.title('Stream Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()







# 15 Additional Application: Quiver plots
fig = plt.figure()
x = np.linspace(-2, 2, 10)
y = np.linspace(-2, 2, 10)
X, Y = np.meshgrid(x, y)
U = -Y
V = X

plt.quiver(X, Y, U, V)
plt.title('Quiver Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()


# 16Additional Application: Contourf plots
fig = plt.figure()

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

plt.contourf(X, Y, Z, cmap='coolwarm')
plt.colorbar(label='Z-value')
plt.title('Contourf Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# 17 Additional Application: Stem plots
fig = plt.figure()

x = np.linspace(0.1, 2 * np.pi, 10)
y = np.cos(x)

plt.stem(x, y, linefmt='b-', markerfmt='ro', basefmt='g-')
plt.title('Stem Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()


# 18 Additional Application: Hexbin plots
fig = plt.figure()

x = np.random.randn(1000)
y = np.random.randn(1000)

plt.hexbin(x, y, gridsize=20, cmap='inferno')
plt.colorbar(label='count in bin')
plt.title('Hexbin Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# 19 Additional Application: Barh plots
categories = ['A', 'B', 'C', 'D']
values = [25, 30, 35, 40]

plt.barh(categories, values, color='skyblue')
plt.title('Horizontal Bar Plot')
plt.xlabel('Values')
plt.ylabel('Categories')
plt.show()



# 20 Additional Application: Stack plots
x = [1, 2, 3, 4, 5]

y1 = [1, 2, 3, 4, 5]
y2 = [2,30, 4, 5, 6]
y3 = [3, 4, 5, 6, 7]
fig = plt.figure()

plt.stackplot(x, y1, y2, y3, labels=['Y1', 'Y2', 'Y3'])
plt.legend(loc='upper left')
plt.title('Stack Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()




# Scikit-learn is a versatile library that can be used to solve a wide range of machine learning problems. Some of the common types of problems that can be addressed with Scikit-learn include:

# 1 Classification: Predicting the category or class label of a given input. Examples include spam detection, image recognition, and sentiment analysis.
# 2 Regression: Predicting continuous numerical values. Examples include house price prediction, stock price forecasting, and demand forecasting.
# 3 Clustering: Grouping similar data points together based on their features. Examples include customer segmentation, anomaly detection, and document clustering.
# 4 Dimensionality Reduction: Reducing the number of features while preserving important information. Examples include visualization of high-dimensional data and feature extraction for machine learning models.
# 5 Model Selection and Evaluation: Comparing different machine learning models and selecting the best one for a given task. Examples include cross-validation, hyperparameter tuning, and model evaluation metrics.
# 6 Preprocessing and Feature Engineering: Transforming and preparing the data before training machine learning models. Examples include feature scaling, missing value imputation, and one-hot encoding.
# 7 Ensemble Methods: Combining multiple machine learning models to improve performance. Examples include random forests, gradient boosting, and stacking.
# 8 Text Mining and Natural Language Processing (NLP): Analyzing and extracting insights from textual data. Examples include text classification, sentiment analysis, and named entity recognition.
# 9 Time Series Analysis: Analyzing and forecasting time-series data. Examples include stock market prediction, weather forecasting, and demand forecasting.
# 10 Anomaly Detection: Identifying unusual patterns or outliers in data. Examples include fraud detection, network intrusion detection, and equipment failure prediction.


# =============================================================================
#   Classification
# =============================================================================

# Importing necessary libraries
from sklearn import datasets  # To import datasets
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.neighbors import KNeighborsClassifier  # To import the KNN classifier
from sklearn import metrics  # To evaluate the model's performance
from sklearn.metrics import confusion_matrix  # To compute the confusion matrix
import matplotlib.pyplot as plt

# 100 
# 70 آموزش
# 20 اعتبار سنجی 
# 10 درصد تست

# plt.figure(figsize=(10, 6))

# Load a dataset (Iris dataset is a commonly used example)
iris = datasets.load_iris()

# Split the dataset into features (X) and target labels (y)
X = iris.data  # Features
y = iris.target  # Target labels

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the KNN classifier with a specified number of neighbors (in this case, 3)
knn = KNeighborsClassifier(n_neighbors = 55)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = knn.predict(X_test)

# Evaluate the performance of the classifier
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

import matplotlib.pyplot as plt
import seaborn as sns

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()








# =============================================================================
# Regression
# =============================================================================
plt.close('all')
# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic data for regression
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Independent variable (feature)
y = 3 * X + 2 + np.random.randn(100, 1) * 2  # Dependent variable (target)

plt.plot(X,y,'*')
# Instantiate and train a linear regression model
model = LinearRegression()
model.fit(X, y)

### sss


# Make predictions on the entire data range
X_range = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = model.predict(X_range)

# Plot the original data and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(X_range, y_pred, color='red', label='Regression Line')
plt.title('Linear Regression Example')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()




# =============================================================================
# Clustering
# =============================================================================
plt.close('all')
# Importing necessary libraries
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))

# Generate synthetic data with 3 clusters
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
X, temp = make_blobs(n_samples=300, centers=3, random_state=42)
# plt.plot(X[:, 0], X[:, 1],'*')
# Instantiate and fit a KMeans clustering model
for i in range (5):
    plt.figure(figsize=(10, 6))
    kmeans = KMeans(n_clusters = i+1 )
    kmeans.fit(X)
    
    # Visualize the clusters
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', color='red', s=300, label='Centroids')
    plt.title('KMeans Clustering')
    plt.xlabel('age')
    plt.ylabel('w(kg)')
    plt.legend()
    plt.show()



plt.close('all')
# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans

# Generate synthetic data for clustering
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# Plot the original data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Perform KMeans clustering with iteration
max_iterations = 10
plt.figure(figsize=(8, 6))
for i in range(1, max_iterations + 1):
    kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=i, n_init=10, random_state=42)
    
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Plot the clustering result for each iteration
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='red')
    plt.title('KMeans Clustering - Iteration {}'.format(i))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    plt.pause(0.5)


plt.close('all')
# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import AgglomerativeClustering

# Generate synthetic data for clustering
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# Plot the original data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Perform Agglomerative Clustering with iteration
max_iterations = 30
plt.figure(figsize=(8, 6))
for i in range(1, max_iterations + 1):
    clustering = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X)
    labels = clustering.labels_
    
    # Plot the clustering result for each iteration
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', alpha=0.5)
    plt.title('Agglomerative Clustering - Iteration {}'.format(i))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    plt.pause(0.5)





# =============================================================================
# Dimensionality Reduction
# =============================================================================

# Importing necessary libraries
plt.close ('all') 
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Instantiate and fit a PCA model
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)

import numpy as np
x= X_reduced[:, 0]
y= X_reduced[:, 1]
z= X_reduced[:, 2]

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x1 = np.random.randn(100)
x2 = np.random.randn(100)
x3= np.random.randn(100)
ax.scatter(x1, x2, x3)
ax.set_title('3D Scatter Plot')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')


# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(X)
# # Visualize the reduced data
# plt.figure(2)
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')

# plt.title('PCA: Iris Dataset')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()



# =============================================================================
# Model Selection and Evaluation
# =============================================================================
import matplotlib.pyplot as plt

# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate a decision tree classifier
clf = DecisionTreeClassifier()

# Perform cross-validation to evaluate the classifier's performance
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)

# Print the mean cross-validation score
print("Mean Cross-Validation Score:", cv_scores.mean())



# =============================================================================
# Preprocessing and Feature Engineering
# =============================================================================

# normalization data
# X = [1 2 5 8 12 5 6]
# صورت کسر
# X2 = [0 1 4 7 11 4 5]
# X3 = [0 1/11 4/11 7/11 1 4/11 5/11]


# X  [ 0 ,  1]     ==>   (X - min)/(max - min)
# X  [ -1 , 1]     ==>   -1+2*(X - min)/(max - min)
# X  [ -1 , 1]     ==>   (X-mean)/var     Z-score





# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
# plt.figure(figsize=(10, 6))

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a preprocessing pipeline
preprocessor = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features by removing the mean and scaling to unit variance
    ('poly_features', PolynomialFeatures(degree=2))  # Generate polynomial features up to degree 2
])

# Fit the preprocessing pipeline on the training data and transform both training and testing data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Instantiate and train a logistic regression classifier
clf = LogisticRegression()
clf.fit(X_train_processed, y_train)

# Evaluate the classifier's accuracy on the processed testing data
accuracy = clf.score(X_test_processed, y_test)
print("Classification Accuracy after Preprocessing:", accuracy)



# =============================================================================
# Ensemble Methods
# =============================================================================

# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# plt.figure(figsize=(10, 6))

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate a Random Forest classifier with 100 decision trees
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Random Forest Classifier:", accuracy)



# =============================================================================
# Text Mining and Natural Language Processing (NLP)
# =============================================================================
# TIME CONSUMER
# Importing necessary libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# plt.figure(figsize=(10, 6))

# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all')

# Split the dataset into features (X) and target labels (y)
X = newsgroups.data
y = newsgroups.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text data into TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Linear Support Vector Classifier (SVM) on the TF-IDF vectors
svm_classifier = LinearSVC()
svm_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_tfidf)

# Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=newsgroups.target_names, yticklabels=newsgroups.target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

###### RESULT IN Group

# =============================================================================
# Time Series Analysis
# =============================================================================
# plt.figure(figsize=(10, 6))

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# Generate synthetic time series data
np.random.seed(42)
X = np.arange(100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(scale=0.1, size=100)

# Instantiate and train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions on the entire time series
y_pred = model.predict(X)

# Plot the original time series data and the predicted values
plt.figure(figsize=(10, 6))
plt.plot(X, y, label='Original Data', color='blue')
plt.plot(X, y_pred, label='Predicted Data', color='red', linestyle='--')
plt.title('Time Series Analysis')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()




plt.figure(figsize=(10, 6))

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Generate synthetic time series data
np.random.seed(42)
X = np.arange(1000).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(scale=0.1, size=1000)

# Instantiate and train a Support Vector Regressor (SVR)
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X, y)

# Make predictions on the entire time series
y_pred = svr.predict(X)

# Plot the original time series data and the predicted values
plt.figure(figsize=(10, 6))
plt.plot(X, y, label='Original Data', color='blue')
plt.plot(X, y_pred, label='Predicted Data', color='red', linestyle='--')
plt.title('Time Series Analysis using SVR')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()



# =============================================================================
# Anomaly Detection
# =============================================================================


# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Generate synthetic time series data with anomalies
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=1000)
anomalies = np.random.normal(loc=5, scale=1, size=20)
data = np.concatenate((normal_data, anomalies))

# Reshape the data for compatibility with Scikit-learn
X = data.reshape(-1, 1)

# Fit an Isolation Forest model for anomaly detection
isolation_forest = IsolationForest(contamination=0.05)
isolation_forest.fit(X)

# Predict outliers/anomalies
y_pred = isolation_forest.predict(X)

# Plot the time series data and highlight anomalies
plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.plot(X, label='Data', color='blue')
plt.scatter(np.where(y_pred == -1), X[y_pred == -1], color='red', marker='x', label='Anomalies')
plt.title('Anomaly Detection using Isolation Forest')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Filter out anomalies from the original data
filtered_data = X[y_pred == 1]

# Plot the time series data after removing anomalies
# plt.figure(figsize=(10, 6))
plt.subplot(122)
plt.plot(filtered_data, label='Data after removing anomalies', color='green')
plt.title('Time Series Data after Removing Anomalies')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()






