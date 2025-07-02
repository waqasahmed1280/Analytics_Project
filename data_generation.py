import pandas as pd
import numpy as np

# Set a seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_samples = 1000

# Feature 1: Age (in years)
age = np.random.randint(18, 65, num_samples)

# Feature 2: Income (in thousands of dollars)
income = np.random.normal(loc=50, scale=15, size=num_samples).astype(int)
income = np.clip(income, 15, 150) # Clip to a reasonable range

# Feature 3: Education level (0: High School, 1: Bachelor's, 2: Master's, 3: PhD)
education = np.random.choice([0, 1, 2, 3], num_samples, p=[0.3, 0.4, 0.2, 0.1])

# Feature 4: Hours worked per week
hours_worked = np.random.normal(loc=40, scale=10, size=num_samples).astype(int)
hours_worked = np.clip(hours_worked, 10, 80)

# Target variable: Likelihood to purchase a product (0: No, 1: Yes)
# Let's make it somewhat dependent on the features
propensity_score = (age / 80) + (income / 150) * 0.5 + (education / 3) * 0.3 + (hours_worked / 80) * 0.2
propensity_score += np.random.normal(0, 0.1, num_samples) # Add some noise
purchase_likelihood = (propensity_score > np.median(propensity_score)).astype(int)


# Create a Pandas DataFrame
data = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Education': education,
    'HoursWorked': hours_worked,
    'Purchase': purchase_likelihood
})

# Save to CSV
file_path = 'synthetic_data.csv'
data.to_csv(file_path, index=False)

print(f"Synthetic data generated and saved to {file_path}")
print(f"Data shape: {data.shape}")
print("\nFirst 5 rows of the data:")
print(data.head())
print("\nTarget variable distribution:")
print(data['Purchase'].value_counts(normalize=True))
