import numpy as np 
import pandas as pd 

# ------------ Data Loading & Cleaning -------------

data = pd.read_csv("E:\Smart Feedback Analyzer\Reviews.csv")
print("==== Shape ====")
print(data.shape)

print("\n==== Columns ====")
print(data.columns.tolist())

print("\n==== Data Types ====")
print(data.dtypes)

print("\n==== Sample Rows ====")
print(data.head())

print("\n=== Missing Values ====")
print(data.isnull().sum())


print("\n=== Duplicates ===")
print(data.duplicated().sum())

print("\n=== Duplictae Text Values ===")
print(data.duplicated(subset=["Text"]).sum())

numerical_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()

print("\n=== Numerical Columns ===")
print(numerical_cols)

print("\n=== Categorical Columns ===")
print(categorical_cols)

print("\n Descriptive Stats (numerical columns)")
print(data[numerical_cols].describe())

print("\n=== Descriptive Stats (categorical columns)")
print(data[categorical_cols].describe())


print("\n=== Score Value Counts ===")
print(data["Score"].value_counts())