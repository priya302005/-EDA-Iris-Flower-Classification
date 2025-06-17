# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the Titanic dataset from the current folder
df = pd.read_csv('train.csv')

# Step 3: Show first 5 rows of the dataset
print(df.head())

print("\nMissing values in each column:")
print(df.isnull().sum())

# Step 3.1: Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Step 3.2: Fill missing 'Age' values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Step 3.3: Fill missing 'Embarked' values with the most frequent value (mode)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Step 3.4: Drop the 'Cabin' column (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

# Step 3.5: Confirm that all missing values are handled
print("\nMissing values after cleaning:")
print(df.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')  # Optional: For clean visuals


sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Gender')
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.show()

sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution of Passengers')
plt.show()

sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age vs Survival')
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

df['Survived'].value_counts(normalize=True) * 100

# ✅ Step 5: Observations Summary (console output)
print("\n📝 Key Insights:")
print("- Most passengers did NOT survive (about 62%).")
print("- Women had a much higher survival rate than men.")
print("- Passengers in 1st class had better survival chances.")
print("- Children (under 10) had a slightly higher survival rate.")
print("- Cabin feature was dropped due to too many missing values.")
