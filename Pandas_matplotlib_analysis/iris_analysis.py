import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load dataset
try:
    iris_raw = load_iris()
    iris_df = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)
    iris_df['species'] = pd.Categorical.from_codes(iris_raw.target, iris_raw.target_names)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Inspect data
print("First 5 rows:")
print(iris_df.head())
print("\nData types and missing values:")
print(iris_df.info())
print(iris_df.isnull().sum())

# Clean data
iris_df.dropna(inplace=True)

# Basic statistics
print("\nDescriptive statistics:")
print(iris_df.describe())

# Group by species
grouped = iris_df.groupby('species').mean()
print("\nMean values by species:")
print(grouped)

# Line chart (simulated time series)
plt.figure(figsize=(10, 5))
plt.plot(iris_df.index, iris_df['sepal length (cm)'], label='Sepal Length')
plt.title('Simulated Time-Series of Sepal Length')
plt.xlabel('Index (as time)')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.grid(True)
plt.show()

# Bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=iris_df, ci=None)
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(iris_df['sepal width (cm)'], bins=20, kde=True)
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=iris_df)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()
