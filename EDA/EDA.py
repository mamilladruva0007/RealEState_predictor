
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

sns.set(rc={'figure.figsize': (10,5)})

df = pd.read_csv(r"C:\Users\mamil\Desktop\RealEstate_App\data\india_housing_prices.csv")


print("\n===== FIRST 10 ROWS =====")
print(df.head(10))

print("\n===== DATA INFO =====")
print(df.info())

print("\n===== SUMMARY STATISTICS =====")
print(df.describe(include="all"))

print("\n===== UNIQUE VALUES PER COLUMN =====")
print(df.nunique())


print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

plt.figure(figsize=(12,5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Value Heatmap")
plt.show()

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())



# 1. Distribution of property prices
sns.histplot(df['Price_in_Lakhs'], kde=True)
plt.title("Price Distribution")
plt.show()

# 2. Distribution of property sizes
sns.histplot(df['Size_in_SqFt'], kde=True)
plt.title("Size Distribution")
plt.show()

# 3. Price per sqft variation by property type
sns.boxplot(x='Property_Type', y='Price_per_SqFt', data=df)
plt.xticks(rotation=45)
plt.title("Price per SqFt by Property Type")
plt.show()

# 4. Relationship between size and price
sns.scatterplot(x='Size_in_SqFt', y='Price_in_Lakhs', hue='BHK', data=df)
plt.title("Size vs Price Relationship")
plt.show()

# 5. Outlier detection
for col in ['Price_in_Lakhs', 'Size_in_SqFt', 'Price_per_SqFt']:
    sns.boxplot(x=df[col])
    plt.title(f"Outliers in {col}")
    plt.show()



# 6. Avg PPS by state
df.groupby('State')['Price_per_SqFt'].mean().sort_values(ascending=False).plot(kind='bar')
plt.title("Avg Price per SqFt by State")
plt.show()

# 7. Avg price by city
df.groupby('City')['Price_in_Lakhs'].mean().sort_values().plot(kind='barh')
plt.title("Avg Property Price by City")
plt.show()

# 8. Median age by locality
df.groupby("Locality")["Age_of_Property"].median().sort_values().head(20).plot(kind='bar')
plt.title("Median Property Age by Locality (Top 20)")
plt.show()

# 9. BHK distribution across cities
sns.countplot(data=df, x='City', hue='BHK')
plt.xticks(rotation=90)
plt.title("BHK Distribution Across Cities")
plt.show()

# 10. Price trends for top 5 costliest localities
top5 = df.groupby("Locality")['Price_per_SqFt'].mean().sort_values(ascending=False).head(5)
print("\n===== TOP 5 COSTLIEST LOCALITIES =====")
print(top5)

top5.plot(kind='bar')
plt.title("Top 5 Expensive Localities by PPS")
plt.show()



# 11. Correlation heatmap
corr = df.select_dtypes(include=['int64','float64']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 12. Nearby schools vs PPS
sns.scatterplot(x='Nearby_Schools', y='Price_per_SqFt', data=df)
plt.title("Nearby Schools vs Price per SqFt")
plt.show()

# 13. Nearby hospitals vs PPS
sns.scatterplot(x='Nearby_Hospitals', y='Price_per_SqFt', data=df)
plt.title("Nearby Hospitals vs Price per SqFt")
plt.show()

# 14. Price vs furnished status
sns.boxplot(x='Furnished_Status', y='Price_in_Lakhs', data=df)
plt.title("Price Variation by Furnished Status")
plt.show()

# 15. PPS vs property facing direction
sns.boxplot(x='Facing', y='Price_per_SqFt', data=df)
plt.title("Price per SqFt by Property Facing")
plt.show()




# 16. Properties per owner type
df['Owner_Type'].value_counts().plot(kind='bar')
plt.title("Owner Type Distribution")
plt.show()

# 17. Availability status distribution
df['Availability_Status'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Availability Status Distribution")
plt.show()

# 18. Parking space effect on price
sns.boxplot(x='Parking_Space', y='Price_in_Lakhs', data=df)
plt.title("Effect of Parking on Price")
plt.show()

# 19. Amenities effect on PPS
df['Amenity_Count'] = df['Amenities'].astype(str).apply(lambda x: len(x.split(',')))
sns.scatterplot(x='Amenity_Count', y='Price_per_SqFt', data=df)
plt.title("Amenities Count vs Price per SqFt")
plt.show()

# 20. Public transport access vs PPS
sns.scatterplot(x='Public_Transport_Accessibility', y='Price_per_SqFt', data=df)
plt.title("Public Transport Access vs PPS")
plt.show()



# 7. DERIVED TARGET: GOOD INVESTMENT FLAG

median_pps = df['Price_per_SqFt'].median()
df['Good_Investment'] = (df['Price_per_SqFt'] <= median_pps).astype(int)

print("\n===== GOOD INVESTMENT VALUE COUNTS =====")
print(df['Good_Investment'].value_counts())

sns.countplot(x='Good_Investment')
plt.title("Good Investment Distribution")
plt.show()


# 8. EXPORT CLEANED DATA

df.to_csv("cleaned_real_estate_data_full.csv", index=False)
print("\n===== CLEANED FILE SAVED: cleaned_real_estate_data_full.csv =====")
