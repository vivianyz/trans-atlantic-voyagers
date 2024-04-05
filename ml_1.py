import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from flask import Flask, request, render_template

# Set the path to the dataverse_files directory
path_to_files = './static/dataverse_files/'

# Load the CSV files into pandas DataFrames
passengers_df = pd.read_csv(f'{path_to_files}ttav_passengers.csv')
voyages_df = pd.read_csv(f'{path_to_files}ttav_voyages.csv')
routes_df = pd.read_csv(f'{path_to_files}ttav_routes.csv')
occupations_df = pd.read_csv(f'{path_to_files}ttav_occupations.csv')

# Perform the joins
passengers_voyages_df = pd.merge(passengers_df, voyages_df, on='MID')
voyages_routes_df = pd.merge(passengers_voyages_df, routes_df, on='routeID')
final_df = pd.merge(voyages_routes_df, occupations_df, on='occID')

# Select only the required columns
final_df = final_df[['MID', 'routeID', 'age', 'sex', 'occID','occ_nm','occ_ctg','occ_grp','port_arv']]

# Encode the 'sex' column if it's categorical. If it's numeric, you can skip this step.
label_encoder = LabelEncoder()
final_df['encoded_sex'] = label_encoder.fit_transform(final_df['sex'].astype(str))

# Prepare features and target variable
X = final_df[['age', 'encoded_sex', 'occID']].copy()  # Work on a copy to avoid changing the original DataFrame
y = final_df['port_arv']

# Impute missing values for 'age' and 'occID' (assuming they are numerical)
# Adjust according to your dataset's needs
imputer = SimpleImputer(strategy='median')
X[['age', 'occID']] = imputer.fit_transform(X[['age', 'occID']])

# Now, we also need to ensure that 'port_arv' is encoded
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
decoded_prediction = label_encoder.inverse_transform(y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(decoded_prediction)
print(classification_report(y_test, y_pred))

# Now, your model is ready to make predictions

# Assuming final_df is your DataFrame and it has a 'port_arv' column

# Count the frequency of each distinct value in 'port_arv'
port_arv_counts = final_df['port_arv'].value_counts()

# Calculate the percentage of each distinct value
port_arv_percentages = final_df['port_arv'].value_counts(normalize=True) * 100

# Combine both counts and percentages into a single DataFrame for easy viewing
port_arv_summary = pd.DataFrame({
    'Count': port_arv_counts,
    'Percentage': port_arv_percentages
})

# Train the Decision Tree Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

joblib.dump(model, 'decision_tree_model.joblib')

# Assuming 'sex' and 'port_arv' are columns in final_df
sex_encoder = LabelEncoder()
port_arv_encoder = LabelEncoder()

# Fit and transform the columns
final_df['sex_encoded'] = sex_encoder.fit_transform(final_df['sex'])
final_df['port_arv_encoded'] = port_arv_encoder.fit_transform(final_df['port_arv'])

# Now, you use 'sex_encoded' and 'port_arv_encoded' for training the model...
# After training, save your model and encoders:
joblib.dump(sex_encoder, 'sex_encoder.joblib')
joblib.dump(port_arv_encoder, 'port_arv_encoder.joblib')






