import numpy as np
import pandas as pd 


data = pd.read_csv("Reviews.csv")
print("Before cleaning:", data.shape)

# ============ Handle Missing Values ===========
data = data.dropna(subset=["ProfileName", "Summary"])
print("After Dropping Missing ProfileName/Summary:",data.shape)

# =========== Remove Duplicates Text =========
data = data.dropna(subset=["Text"])
print("After Removing Duplicates Text:", data.shape)

# ========= Verify The Data Types ==========
data["Score"] = data["Score"].astype(int)
data["Time"] = pd.to_datetime(data["Time"], unit="s")
print("\n--- Data Types After Fixing ---")
print(data.dtypes)

# ========= Check Invalid/garbage rows =======
# print("\nScore Range Check:", data["Score"].min(), "to", data["Score"].max())

# data = data[data["Text"].str.len() > 5]
# print("After Removing too-short Reviews:", data.shape)
# data_final = data[["Score", "Text", "Summary", "Time"]]
# sample = data_final.sample(n=15000, random_state=42)

# sample.to_csv("data/reviews_sample.csv", index=False)
# print("\nFinal Cleaned Smaple Saved! Shape:", sample.shape)
# print(sample.dtypes)


print("\nScore range check:", data["Score"].min(), "to", data["Score"].max())

# Text bohot chhota (jaise 1-2 letters) to wo bhi garbage hai
df = data[data["Text"].str.len() > 5]
print("After removing too-short reviews:", data.shape)

# STEP 5: AB (aur sirf ab) zaroori columns select karo — cleaning ke baad
data_final = data[["Score", "Time", "Summary", "Text"]]

# STEP 6: Sample nikaalo (15000 rows) taake fast chale
sample = data_final.sample(n=15000, random_state=42)

sample.to_csv("data/reviews_sample.csv", index=False)
print("\nFinal cleaned sample saved! Shape:", sample.shape)
print(sample.dtypes)