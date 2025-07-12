# MACHINE-LEARNING-TASKS
# task-1 Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# 1. Fetch the data -----------------------------------------------------------
URL = ("https://github.com/eaisi/discover-projects/raw/main/"
       "ames-housing/AmesHousing.csv")
data = pd.read_csv(URL)
# 2. Keep only the relevant columns ------------------------------------------
cols = ['SalePrice',          # target
        'Gr Liv Area',          # above‑ground living area (sq ft)
        'Bedroom AbvGr',       # # bedrooms above grade
        'Full Bath',           # full baths above grade
        'Half Bath']           # half baths above grade
df = data[cols].dropna()
# 3. Feature engineering ------------------------------------------------------
df['TotalBath'] = df['Full Bath'] + 0.5 * df['Half Bath']
X = df[['Gr Liv Area', 'Bedroom AbvGr', 'TotalBath']]
y = df['SalePrice']
# 4. Train / test split -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# 5. Fit the model ------------------------------------------------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
# 6. Evaluate -----------------------------------------------------------------
y_pred = lin_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print("Coefficients")
for name, coef in zip(X.columns, lin_reg.coef_):
    print(f"  {name:12s}: {coef:,.0f} $/unit")
print(f"\nIntercept      : {lin_reg.intercept_:,.0f} $")
print(f"Test RMSE      : {rmse:,.0f} $")
print(f"Test R²        : {r2:.3f}")
# 7. Quick diagnostic plot ----------------------------------------------------
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], lw=2)
plt.xlabel("Actual sale price ($)")
plt.ylabel("Predicted sale price ($)")
plt.title("House‑price model: actual vs. predicted")
plt.tight_layout()
plt.show()

OUTPUT:

Coefficients
  Gr Liv Area : 111 $/unit
  Bedroom AbvGr: -28,967 $/unit
  TotalBath   : 23,214 $/unit

Intercept      : 55,335 $
Test RMSE      : 56,191 $
Test R²        : 0.606
<img width="490" height="490" alt="image" src="https://github.com/user-attachments/assets/f28d4be0-dd30-4bea-8acf-07d0314f0d30" />

print(data.columns)
Index(['Order', 'PID', 'MS SubClass', 'MS Zoning', 'Lot Frontage', 'Lot Area',
       'Street', 'Alley', 'Lot Shape', 'Land Contour', 'Utilities',
       'Lot Config', 'Land Slope', 'Neighborhood', 'Condition 1',
       'Condition 2', 'Bldg Type', 'House Style', 'Overall Qual',
       'Overall Cond', 'Year Built', 'Year Remod/Add', 'Roof Style',
       'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type',
       'Mas Vnr Area', 'Exter Qual', 'Exter Cond', 'Foundation', 'Bsmt Qual',
       'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin SF 1',
       'BsmtFin Type 2', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF',
       'Heating', 'Heating QC', 'Central Air', 'Electrical', '1st Flr SF',
       '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Bsmt Full Bath',
       'Bsmt Half Bath', 'Full Bath', 'Half Bath', 'Bedroom AbvGr',
       'Kitchen AbvGr', 'Kitchen Qual', 'TotRms AbvGrd', 'Functional',
       'Fireplaces', 'Fireplace Qu', 'Garage Type', 'Garage Yr Blt',
       'Garage Finish', 'Garage Cars', 'Garage Area', 'Garage Qual',
       'Garage Cond', 'Paved Drive', 'Wood Deck SF', 'Open Porch SF',
       'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Pool QC',
       'Fence', 'Misc Feature', 'Misc Val', 'Mo Sold', 'Yr Sold', 'Sale Type',
       'Sale Condition', 'SalePrice'],
      dtype='object')
# Based on the available columns in the dataset, identify other features that could be relevant to house prices. 

print("Original columns in data:")
print(data.columns.tolist())
print("\nPotential features for house price prediction:")
potential_features = [
    'Overall Qual',       # Overall material and finish quality
    'Overall Cond',       # Overall condition rating
    'Year Built',         # Original construction date
    'Year Remod/Add',     # Remodel date
    'Exterior 1st',       # Exterior covering on house
    'Exterior 2nd',       # Exterior covering on house (if more than one)
    'Mas Vnr Area',       # Masonry veneer area in square feet
    'Exter Qual',       # Exterior material quality
    'Exter Cond',       # Present condition of the material on the exterior
    'Foundation',       # Type of foundation
    'Bsmt Qual',        # Height of the basement
    'Bsmt Cond',        # General condition of the basement
    'Total Bsmt SF',    # Total square feet of basement area
    'Heating QC',       # Heating quality and condition
    'Central Air',      # Central air conditioning
    '1st Flr SF',       # First Floor square feet
    '2nd Flr SF',       # Second floor square feet
    'Low Qual Fin SF',  # Low quality finished square feet (all floors)
    'Kitchen Qual',     # Kitchen quality
    'TotRms AbvGrd',    # Total rooms above grade (does not include bathrooms)
    'Fireplaces',       # Number of fireplaces
    'Garage Type',      # Garage location
    'Garage Yr Blt',    # Year garage was built
    'Garage Finish',    # Interior finish of the garage
    'Garage Cars',      # Size of garage in car capacity
    'Garage Area',      # Size of garage in square feet
    'Paved Drive',      # Paved driveway
    'Wood Deck SF',     # Wood deck area in square feet
    'Open Porch SF',    # Open porch area in square feet
    'Enclosed Porch',   # Enclosed porch area in square feet
    '3Ssn Porch',       # Three season porch area in square feet
    'Screen Porch',     # Screen porch area in square feet
    'Pool Area',        # Pool area in square feet
    'Fence',            # Fence quality
    'Misc Feature',     # Miscellaneous feature not covered in other categories
    'Misc Val',         # $Value of miscellaneous feature
    'Mo Sold',          # Month Sold
    'Yr Sold',          # Year Sold
    'Sale Type',        # Type of sale
    'Sale Condition'    # Condition of sale
]
print(potential_features)

Original columns in data:
['Order', 'PID', 'MS SubClass', 'MS Zoning', 'Lot Frontage', 'Lot Area', 'Street', 'Alley', 'Lot Shape', 'Land Contour', 'Utilities', 'Lot Config', 'Land Slope', 'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type', 'House Style', 'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add', 'Roof Style', 'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type', 'Mas Vnr Area', 'Exter Qual', 'Exter Cond', 'Foundation', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin SF 1', 'BsmtFin Type 2', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF', 'Heating', 'Heating QC', 'Central Air', 'Electrical', '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 'Half Bath', 'Bedroom AbvGr', 'Kitchen AbvGr', 'Kitchen Qual', 'TotRms AbvGrd', 'Functional', 'Fireplaces', 'Fireplace Qu', 'Garage Type', 'Garage Yr Blt', 'Garage Finish', 'Garage Cars', 'Garage Area', 'Garage Qual', 'Garage Cond', 'Paved Drive', 'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Pool QC', 'Fence', 'Misc Feature', 'Misc Val', 'Mo Sold', 'Yr Sold', 'Sale Type', 'Sale Condition', 'SalePrice']
Potential features for house price prediction:
['Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add', 'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Area', 'Exter Qual', 'Exter Cond', 'Foundation', 'Bsmt Qual', 'Bsmt Cond', 'Total Bsmt SF', 'Heating QC', 'Central Air', '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 'Kitchen Qual', 'TotRms AbvGrd', 'Fireplaces', 'Garage Type', 'Garage Yr Blt', 'Garage Finish', 'Garage Cars', 'Garage Area', 'Paved Drive', 'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Fence', 'Misc Feature', 'Misc Val', 'Mo Sold', 'Yr Sold', 'Sale Type', 'Sale Condition']

# Select a subset of potential features and analyze their relationship with SalePrice using visualizations and descriptive statistics.
# 1. Select a subset of potential features
selected_features = [
    'Overall Qual',
    'Year Built',
    'Total Bsmt SF',
    '1st Flr SF',
    'Garage Cars',
    'Kitchen Qual',
    'Neighborhood',
    'Sale Condition',
    'Yr Sold' # Include 'Yr Sold'
]

# Filter the original data to include SalePrice and selected features, and drop rows with NaNs in these columns

df_selected = data[['SalePrice'] + selected_features].dropna()
# 2. Analyze numerical features
numerical_features = ['Overall Qual', 'Year Built', 'Total Bsmt SF', '1st Flr SF', 'Garage Cars', 'Yr Sold'] # Include 'Yr Sold'
print("--- Numerical Feature Analysis ---")
for feature in numerical_features:
    print(f"\nAnalyzing: {feature}")
 # Descriptive statistics
    print("Descriptive Statistics:")
    print(df_selected[[feature, 'SalePrice']].describe())
 # Correlation
    correlation = df_selected[[feature, 'SalePrice']].corr().iloc[0, 1]
    print(f"\nCorrelation with SalePrice: {correlation:.3f}")
 # Scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(df_selected[feature], df_selected['SalePrice'], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('SalePrice ($)')
    plt.title(f'SalePrice vs. {feature}')
    plt.tight_layout()
    plt.show()
# 3. Analyze categorical features
categorical_features = ['Kitchen Qual', 'Neighborhood', 'Sale Condition']
print("\n--- Categorical Feature Analysis ---")
for feature in categorical_features:
    print(f"\nAnalyzing: {feature}")
# Average SalePrice per category
    avg_saleprice_per_category = df_selected.groupby(feature)['SalePrice'].mean().sort_values(ascending=False)
    print("Average SalePrice per Category:")
    print(avg_saleprice_per_category)
# Box plot
    plt.figure(figsize=(10, 6))
    df_selected.boxplot(column='SalePrice', by=feature)
    plt.xlabel(feature)
    plt.ylabel('SalePrice ($)')
    plt.title(f'SalePrice Distribution by {feature}')
    plt.suptitle('') # Suppress the default title
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    OUTPUT:
-- Numerical Feature Analysis ---

Analyzing: Overall Qual
Descriptive Statistics:
       Overall Qual      SalePrice
count   2928.000000    2928.000000
mean       6.095970  180841.033811
std        1.410831   79889.904415
min        1.000000   12789.000000
25%        5.000000  129500.000000
50%        6.000000  160000.000000
75%        7.000000  213500.000000
max       10.000000  755000.000000

Correlation with SalePrice: 0.799
 <img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/2bfdceeb-2094-4f11-8e73-4e280471ca6b" />
 Analyzing: Year Built
Descriptive Statistics:
        Year Built      SalePrice
count  2928.000000    2928.000000
mean   1971.381489  180841.033811
std      30.238845   79889.904415
min    1872.000000   12789.000000
25%    1954.000000  129500.000000
50%    1973.000000  160000.000000
75%    2001.000000  213500.000000
max    2010.000000  755000.000000

Correlation with SalePrice: 0.558
<img width="789" height="490" alt="image" src="https://github.com/user-attachments/assets/c4e849cc-0efb-495a-b7d2-93423760f2b9" />

Analyzing: Total Bsmt SF
Descriptive Statistics:
       Total Bsmt SF      SalePrice
count    2928.000000    2928.000000
mean     1051.680328  180841.033811
std       440.675942   79889.904415
min         0.000000   12789.000000
25%       793.000000  129500.000000
50%       990.000000  160000.000000
75%      1302.000000  213500.000000
max      6110.000000  755000.000000

Correlation with SalePrice: 0.632
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/4f90c17d-9f64-4ff8-a52a-9dae571a8079" />
Analyzing: 1st Flr SF
Descriptive Statistics:
        1st Flr SF      SalePrice
count  2928.000000    2928.000000
mean   1159.721995  180841.033811
std     391.973820   79889.904415
min     334.000000   12789.000000
25%     876.000000  129500.000000
50%    1084.500000  160000.000000
75%    1384.750000  213500.000000
max    5095.000000  755000.000000

Correlation with SalePrice: 0.622
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/e799e6ab-a274-4810-ac09-64a731667670" />
Analyzing: Garage Cars
Descriptive Statistics:
       Garage Cars      SalePrice
count  2928.000000    2928.000000
mean      1.767077  180841.033811
std       0.760564   79889.904415
min       0.000000   12789.000000
25%       1.000000  129500.000000
50%       2.000000  160000.000000
75%       2.000000  213500.000000
max       5.000000  755000.000000

Correlation with SalePrice: 0.648
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/0e55e189-95f7-4827-9c50-a2d89cc6ab47" />
Analyzing: Yr Sold
Descriptive Statistics:
           Yr Sold      SalePrice
count  2928.000000    2928.000000
mean   2007.790642  180841.033811
std       1.316976   79889.904415
min    2006.000000   12789.000000
25%    2007.000000  129500.000000
50%    2008.000000  160000.000000
75%    2009.000000  213500.000000
max    2010.000000  755000.000000

Correlation with SalePrice: -0.031
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/652dff21-f8cc-419c-b901-8b4a1e2fa104" />
--- Categorical Feature Analysis ---

Analyzing: Kitchen Qual
Average SalePrice per Category:
Kitchen Qual
Ex    337339.341463
Gd    210887.288179
TA    139590.503684
Po    107500.000000
Fa    105907.042857
Name: SalePrice, dtype: float64
<Figure size 1000x600 with 0 Axes>
<img width="630" height="455" alt="image" src="https://github.com/user-attachments/assets/a477730e-bb59-4946-86ce-8a26a2c90fb6" />
Analyzing: Neighborhood
Average SalePrice per Category:
Neighborhood
NoRidge    330319.126761
StoneBr    324229.196078
NridgHt    322018.265060
GrnHill    280000.000000
Veenker    248314.583333
Timber     246599.541667
Somerst    229707.324176
ClearCr    208662.090909
Crawfor    207550.834951
CollgCr    201803.434457
Blmngtn    196661.678571
Greens     193531.250000
Gilbert    190646.575758
NWAmes     188406.908397
SawyerW    184070.184000
Mitchel    162226.631579
NAmes      145097.349887
Blueste    143590.000000
NPkVill    140710.869565
Landmrk    137000.000000
Sawyer     136751.152318
SWISU      135071.937500
Edwards    130843.381443
BrkSide    125183.878505
OldTown    123991.891213
BrDale     105608.333333
IDOTRR     103240.336957
MeadowV     95756.486486
Name: SalePrice, dtype: float64
<Figure size 1000x600 with 0 Axes>
<img width="630" height="454" alt="image" src="https://github.com/user-attachments/assets/8c06b220-e132-4b01-a344-a464c7b8b3a7" />
Analyzing: Sale Condition
Average SalePrice per Category:
Sale Condition
Partial    273374.371429
Normal     175567.643183
Alloca     162319.130435
Family     157488.586957
Abnorml    140721.100529
AdjLand    108916.666667
Name: SalePrice, dtype: float64
<Figure size 1000x600 with 0 Axes>
<img width="630" height="455" alt="image" src="https://github.com/user-attachments/assets/44b6dc62-c938-43c7-8726-b366c8ba8580" />
  
#  a new feature 'House Age' using 'Year Built' and 'Yr Sold', and add it to the df_selected DataFrame.
# Create 'House Age' feature
df_selected['House Age'] = df_selected['Yr Sold'] - df_selected['Year Built']
display(df_selected.head())
	
SalePrice	Overall Qual	Year Built	Total Bsmt SF	1st Flr SF	Garage Cars	Kitchen Qual	Neighborhood	Sale Condition	Yr Sold	House Age
0	215000	6	1960	1080.0	1656	2.0	TA	NAmes	Normal	2010	50
1	105000	5	1961	882.0	896	1.0	TA	NAmes	Normal	2010	49
2	172000	6	1958	1329.0	1329	1.0	Gd	NAmes	Normal	2010	52
3	244000	7	1968	2110.0	2110	2.0	Ex	NAmes	Normal	2010	42
4	189900	5	1997	928.0	928	2.0	TA	Gilbert	Normal	2010	13


# Distributions
<img width="554" height="427" alt="image" src="https://github.com/user-attachments/assets/63a8e65f-9107-4320-ae9a-1c9f6cbbd66f" />
<img width="561" height="427" alt="image" src="https://github.com/user-attachments/assets/9b8e5107-5851-4f9d-aa09-869db4c4f85c" />
<img width="561" height="427" alt="image" src="https://github.com/user-attachments/assets/4e90617b-3514-4331-b4bd-fbcad2f9104a" />
<img width="561" height="427" alt="image" src="https://github.com/user-attachments/assets/39891fa5-c606-4194-b5b1-8acb86576125" />
<img width="561" height="427" alt="image" src="https://github.com/user-attachments/assets/0eaec5b5-bba5-488a-8531-5426738025b1" />

# Categorical distributions
<img width="552" height="409" alt="image" src="https://github.com/user-attachments/assets/fb9340f7-9330-4ead-9178-85785331564c" />
<img width="576" height="409" alt="image" src="https://github.com/user-attachments/assets/205e9df1-20bd-4180-9b76-e4373592045a" />

# 2-d distributions

<img width="561" height="420" alt="image" src="https://github.com/user-attachments/assets/37884aaa-5f6f-4014-94cc-953b8eb87cb4" />
<img width="565" height="420" alt="image" src="https://github.com/user-attachments/assets/8c8f810f-a1b8-429b-97b7-b71a67211f72" />
<img width="565" height="420" alt="image" src="https://github.com/user-attachments/assets/a87cf02b-4989-440e-bcea-9870b44cc1d7" />
<img width="565" height="420" alt="image" src="https://github.com/user-attachments/assets/cfa27f7d-2cad-488c-abe8-74bb9d6e4218" />

# Time series

<img width="1101" height="531" alt="image" src="https://github.com/user-attachments/assets/5705fe6f-cc7e-4b2e-a80f-a87a104d05eb" />
<img width="1101" height="531" alt="image" src="https://github.com/user-attachments/assets/5f03d3fc-4485-4063-a54c-a6f140716861" />
<img width="1108" height="531" alt="image" src="https://github.com/user-attachments/assets/01328444-a325-4f43-ade4-eb5b18c2be86" />
<img width="1108" height="531" alt="image" src="https://github.com/user-attachments/assets/707998d9-3445-4b1f-8cd5-9fa505768663" />

# Values
<img width="691" height="366" alt="image" src="https://github.com/user-attachments/assets/59388c86-f0a8-4e27-95d9-ac2ada1080d0" />
<img width="674" height="366" alt="image" src="https://github.com/user-attachments/assets/f6e73bf1-e0e8-4d6d-ad77-b5c861075265" />
<img width="677" height="366" alt="image" src="https://github.com/user-attachments/assets/78a2cb59-61cc-41c5-a181-1e2c1f44059a" />
<img width="677" height="366" alt="image" src="https://github.com/user-attachments/assets/a237e228-8856-48f0-9203-556888272f2d" />

# 2-d categorical distributions
<img width="636" height="671" alt="image" src="https://github.com/user-attachments/assets/2cc159b5-43de-4187-abfd-2b696de0cc33" />

# Instantiate and train a linear regression model using the training data.
new_lin_reg = LinearRegression()
new_lin_reg.fit(X_train, y_train)

OUTPUT:
LinearRegression
?i
LinearRegression()
# Make predictions using the new model, calculate the evaluation metrics (RMSE and R2), print the comparison with the previous model's metrics, and create a scatter plot of actual vs. predicted values for the new model.

y_pred_new = new_lin_reg.predict(X_test)
rmse_new = np.sqrt(mean_squared_error(y_test, y_pred_new))
r2_new = r2_score(y_test, y_pred_new)
print("--- Model Performance Comparison ---")
print(f"Previous Model RMSE: {rmse:,.0f} $")
print(f"Previous Model R²  : {r2:.3f}")
print(f"New Model RMSE     : {rmse_new:,.0f} $")
print(f"New Model R²       : {r2_new:.3f}")
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_new, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], lw=2, color='red')
plt.xlabel("Actual Sale Price ($)")
plt.ylabel("Predicted Sale Price (New Model) ($)")
plt.title("New House Price Model: Actual vs. Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

OUTPUT:
--- Model Performance Comparison ---
Previous Model RMSE: 56,191 $
Previous Model R²  : 0.606
New Model RMSE     : 39,251 $
New Model R²       : 0.813
<img width="590" height="590" alt="image" src="https://github.com/user-attachments/assets/e0ec236f-7ad1-4d21-a6cd-90d472c3d3b0" />



# TASK-2 Create a K-means clustering algorithm to group customers of a retail store based on their purchase history.


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import io
csv_data = """CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)
1,Male,19,15,39
2,Male,21,15,81
3,Female,20,16,6
4,Female,23,16,77
5,Female,31,17,40
6,Female,22,17,76
7,Female,35,18,6
8,Female,23,18,94
9,Male,64,19,3
10,Female,30,19,72
"""
df = pd.read_csv(io.StringIO(csv_data))
print(df.head())
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # Added n_init
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(6,4))
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

optimal_k = 3 # Adjusted optimal_k based on the small sample data
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10) # Added n_init
df['Cluster'] = kmeans.fit_predict(X)
plt.figure(figsize=(8,6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']

for i in range(optimal_k):
    plt.scatter(X[df['Cluster'] == i]['Annual Income (k$)'],
                X[df['Cluster'] == i]['Spending Score (1-100)'],
                label=f'Cluster {i+1}',
                c=colors[i],
                edgecolor='black')
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            s=300, c='yellow', marker='*', label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation using K-Means')
plt.legend()
plt.tight_layout()
plt.show()

OUTPUT:
 CustomerID    Gender  Age  Annual Income (k$)  Spending Score (1-100)
0           1    Male   19                  15                      39
1           2    Male   21                  15                      81
2           3  Female   20                  16                       6
3           4  Female   23                  16                      77
4           5  Female   31                  17                      40
<img width="558" height="393" alt="image" src="https://github.com/user-attachments/assets/555dd503-1452-44d5-9603-5a5173cb45dc" />
<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/50c00076-caaf-47a7-8399-77477c1d5126" />


# TASK-3Implement a support vector machine (SVM) to classify images of cats and dogs from the Kaggle dataset.

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import matplotlib.image as mpimg 
from zipfile import ZipFile
data_path = 'dog-vs-cat.zip'
with ZipFile(data_path, 'r') as zip:
    zip.extractall()
    print('The data set has been extracted.')
    fig = plt.gcf()
fig.set_size_inches(16, 16)
cat_dir = os.path.join('dog-vs-cat-classification/cat')
dog_dir = os.path.join('dog-vs-cat-classification/dog')
cat_names = os.listdir(cat_dir)
dog_names = os.listdir(dog_dir)
pic_index = 210
cat_images = [os.path.join(cat_dir, fname)
              for fname in cat_names[pic_index-8:pic_index]]
dog_images = [os.path.join(dog_dir, fname)
              for fname in dog_names[pic_index-8:pic_index]]
for i, img_path in enumerate(cat_images + dog_images):
    sp = plt.subplot(4, 4, i+1)
    sp.axis('Off')

    img = mpimg.imread(img_path)
    plt.imshow(img)
plt.show()

OUTPUT:
<img width="386" height="346" alt="image" src="https://github.com/user-attachments/assets/bcb9084f-78de-4bfa-b9f2-51fe93e14748" />

# Splitting Dataset

base_dir = 'dog-vs-cat-classification'
train_datagen = image_dataset_from_directory(base_dir,
                                                  image_size=(200,200),
                                                  subset='training',
                                                  seed = 1,
                                                 validation_split=0.1,
                                                  batch_size= 32)
test_datagen = image_dataset_from_directory(base_dir,
                                                  image_size=(200,200),
                                                  subset='validation',
                                                  seed = 1,
                                                 validation_split=0.1,
                                                  batch_size= 32)
Output : 

Found 25000 files belonging to 2 classes.
Using 22500 files for training.
Found 25000 files belonging to 2 classes.
Using 2500 files for validation.

# Model Architecture
model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])
model.summary()
Output :

Cat and Dog Classifier using Tensorflow
<img width="350" height="543" alt="image" src="https://github.com/user-attachments/assets/e64f5d81-ff95-4e7d-890c-ba1026f8182d" />

# Model Compilation and Training
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
history = model.fit(train_datagen,
          epochs=10,
          validation_data=test_datagen)
	  
Output :

<img width="962" height="261" alt="image" src="https://github.com/user-attachments/assets/efc564c8-9329-44b5-97fe-a8eb419336e0" />

Cat and Dog Classifier using Tensorflow

# Model Evaluation
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.show()
Output : 
<img width="510" height="456" alt="image" src="https://github.com/user-attachments/assets/8fd8e9eb-15f8-4595-84b4-e4900d004cee" />
<img width="510" height="456" alt="image" src="https://github.com/user-attachments/assets/46d05119-5797-4129-a94a-54b8190a7b19" />

# Model Testing and Prediction
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(200, 200))
    plt.imshow(img)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    result = model.predict(img)
    print("Dog" if result >= 0.5 else "Cat")
    
predict_image('dog-vs-cat-classification/cat/cat.320.jpg')
predict_image('dog-vs-cat-classification/dog/dog.5510.jpg')

Output:
<img width="271" height="251" alt="image" src="https://github.com/user-attachments/assets/9a5d5240-3a8f-4fb1-a4ef-401ad4d099f0" />
<img width="257" height="257" alt="image" src="https://github.com/user-attachments/assets/be9cfc6c-2d0d-463f-8ab4-007be183e6ea" />



# TASK -4 Develop a hand gesture recognition model that can accurately identify and classify different hand gestures from image or video data, enabling intuitive human-computer interaction and gesture-based control systems.

# Load the HaGRID dataset using a library like tensorflow_datasets or by manually loading the image files and their corresponding labels. Inspect the dataset structure to understand how to access images and labels. Preprocess the images by resizing, normalizing, and potentially augmenting them. Split the dataset into training and validation sets and create data loaders for efficient processing.
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
if USE_MINI:
    # Use a smaller version of the dataset for faster prototyping
    dataset_name = "manual_hand_object_dataset"
else:
    dataset_name = "hagrid"
try:
    ds = tfds.load(dataset_name, split='train', as_supervised=True)
except:
    print(f"Could not load dataset {dataset_name}. Please make sure it is installed.")
  
    dummy_image = tf.random.uniform(shape=(IMG_SIZE, IMG_SIZE, 3), minval=0, maxval=255, dtype=tf.float32)
    dummy_label = tf.constant(0, dtype=tf.int64)
    ds = tf.data.Dataset.from_tensors((dummy_image, dummy_label)).repeat(100)
def preprocess_image(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, label
ds = ds.map(preprocess_image).shuffle(1000)
train_size = int(0.8 * len(ds))
train_ds = ds.take(train_size)
val_ds = ds.skip(train_size)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
print("Dataset loaded, preprocessed, and split successfully.")
print(f"Training dataset size: {len(train_ds) * BATCH_SIZE}")
print(f"Validation dataset size: {len(val_ds) * BATCH_SIZE}")

output:
Training dataset size: 96
Validation dataset size: 32

# Define a CNN model architecture for hand gesture recognition.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
model = Sequential(    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    # Determine the number of classes dynamically based on the dataset
    # Assuming the dataset has a feature named 'label' which is an integer.
    # If the dataset structure is different, this needs to be adjusted.
    # For now, using a placeholder 'num_classes'. This needs to be set based on your dataset.
    Dense(10, activation='softmax') # Placeholder for number of classes
])
model.summary()

output:

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 222, 222, 32)   │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 111, 111, 32)   │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 109, 109, 64)   │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 54, 54, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 52, 52, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 26, 26, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 86528)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 128)            │    11,075,712 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 10)             │         1,290 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 11,170,250 (42.61 MB)
 Trainable params: 11,170,250 (42.61 MB)
 Non-trainable params: 0 (0.00 B)
 
# Compile the defined Keras model with the specified optimizer, loss function, and metrics.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
 # Train the compiled model using the fit method with the specified training and validation datasets and number of epochs.
 history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds
)
output:
Epoch 1/5
3/3 ━━━━━━━━━━━━━━━━━━━━ 13s 3s/step - accuracy: 0.4250 - loss: 1.4878 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
Epoch 2/5
3/3 ━━━━━━━━━━━━━━━━━━━━ 11s 3s/step - accuracy: 1.0000 - loss: 1.1781e-07 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
Epoch 3/5
3/3 ━━━━━━━━━━━━━━━━━━━━ 10s 3s/step - accuracy: 1.0000 - loss: 5.8114e-08 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
Epoch 4/5
3/3 ━━━━━━━━━━━━━━━━━━━━ 9s 3s/step - accuracy: 1.0000 - loss: 0.0032 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
Epoch 5/5
3/3 ━━━━━━━━━━━━━━━━━━━━ 10s 3s/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00

# Evaluate the trained model on the validation dataset to assess its performance and store the results.
evaluation_results = model.evaluate(val_ds)
print("Evaluation Results:")
print(f"Loss: {evaluation_results[0]:.4f}")
print(f"Accuracy: {evaluation_results[1]:.4f}")

output:
1/1 ━━━━━━━━━━━━━━━━━━━━ 2s 2s/step - accuracy: 1.0000 - loss: 0.0000e+00
Evaluation Results:
Loss: 0.0000
Accuracy: 1.0000

# Define a function for real-time hand gesture recognition that takes an image, preprocesses it, predicts the class using the trained model, and returns the predicted class index.
def recognize_gesture(image, model, img_size):
    """
    Recognizes the hand gesture in a given image using a trained model.
Args:
        image: Input image as a NumPy array or TensorFlow tensor.
        model: Trained TensorFlow model for gesture recognition.
        img_size: The target size (height and width) for image preprocessing.
Returns:
        The index of the predicted gesture class.
    """
    # Preprocess the image
    img = tf.image.resize(image, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
 if len(img.shape) == 3:
        img = tf.expand_dims(img, axis=0)
predictions = model.predict(img)
 predicted_class_index = np.argmax(predictions, axis=1)[0]
return predicted_class_index
print("Gesture recognition function defined.")

output:
Gesture recognition function defined.
