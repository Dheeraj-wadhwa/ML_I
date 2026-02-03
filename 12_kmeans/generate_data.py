import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 200
customer_ids = np.arange(1, n_samples + 1)
genders = np.random.choice(['Male', 'Female'], n_samples)
ages = np.random.randint(18, 70, n_samples)
income = np.random.randint(15, 140, n_samples)
# Simulate some clusters for spending score
spending_score = np.zeros(n_samples)
for i in range(n_samples):
    if income[i] > 100:
        spending_score[i] = np.random.randint(1, 40) if np.random.rand() > 0.5 else np.random.randint(60, 100)
    elif income[i] < 40:
        spending_score[i] = np.random.randint(1, 40) if np.random.rand() > 0.5 else np.random.randint(60, 100)
    else:
        spending_score[i] = np.random.randint(40, 60)

data = {
    'CustomerID': customer_ids,
    'Gender': genders,
    'Age': ages,
    'Annual Income (k$)': income,
    'Spending Score (1-100)': spending_score.astype(int)
}

df = pd.DataFrame(data)
df.to_csv('Mall_Customers.csv', index=False)
print("Synthetic 'Mall_Customers.csv' generated successfully.")
