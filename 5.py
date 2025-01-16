import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# Load the datasets
data = pd.read_csv('Dev_data_to_be_shared.csv')

# Preprocess the training dataset
data = data.fillna(0)  # Fill missing values with 0

X = data.drop(['account_number', 'bad_flag'],axis=1)# INPUT
y = data['bad_flag'] # TARGET

# Preprocess the test dataset (exclude bad_flag since it's not present) # Fill missing values with 0
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print('resampled the data')


# Split the training data into training and validation sets (if needed)
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Create a pipeline with scaling, PCA, and logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.90)),  # Keep 95% of the variance
    ('model', LogisticRegression(solver='lbfgs',max_iter=1000,class_weight='balanced'))
])
print('pipeline created')
# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Save the pipeline (preprocessing + PCA + model)
joblib.dump(pipeline, 'model_with_pca_pipeline.pkl')
print("Pipeline saved as 'model_with_pca_pipeline.pkl'")

# Evaluate the pipeline on the validation set
y_val_pred = pipeline.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy}")
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# Check the explained variance ratio
explained_variance = pipeline.named_steps['pca'].explained_variance_ratio_
print(f"Explained variance ratio: {explained_variance}")
print(f"Total variance explained by the selected components: {sum(explained_variance):.2f}")
for i, var in enumerate(explained_variance):
    print(f"Principal Component {i+1}: {var*100:.2f}% of the variance")
