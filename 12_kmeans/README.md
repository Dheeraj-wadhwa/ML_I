# Mall Customer Segmentation using KMeans

This project performs customer segmentation on the Mall Customer dataset using the KMeans clustering algorithm. It categorizes customers into distinct groups based on their **Annual Income** and **Spending Score** to provide actionable business insights.

## 🚀 Project Overview

The project follows a standard machine learning pipeline:
1.  **Data Generation/Loading**: Uses synthetic or Kaggle's Mall Customer dataset.
2.  **Exploratory Data Analysis (EDA)**: Inspection of feature distributions and missing values.
3.  **Data Preprocessing**: Feature scaling using `StandardScaler`.
4.  **Optimal K Determination**: Using the **Elbow Method** to find the inertia "knee".
5.  **KMeans Clustering**: Training the final model with K=5.
6.  **Visualization & Interpretation**: Plotting the resulting clusters and labeling them for business use.

## 📊 Customer Segments

The following segments were identified:
-   **Premium Customers**: High Income, High Spending (Target for loyalty programs).
-   **Careful Customers**: High Income, Low Spending (Target for value-based offers).
-   **Impulsive Customers**: Low Income, High Spending (Likely younger audience).
-   **Budget Customers**: Low Income, Low Spending.
-   **Standard Customers**: Average Income and Spending.

## 📁 Repository Structure

-   `customer_segmentation.py`: The main clustering logic and pipeline.
-   `generate_data.py`: A helper script to create synthetic data if the source CSV is missing.
-   `Mall_Customers.csv`: The input dataset.
-   `segmented_customers.csv`: The output dataset containing cluster IDs and labels.
-   `elbow_curve.png`: Plot visualization for the Elbow Method.
-   `customer_clusters.png`: Scatter plot visualization of the final clusters.

## 🛠️ Requirements

-   Python 3.10
-   Pandas, NumPy
-   Scikit-Learn
-   Matplotlib, Seaborn

## 🏃 How to Run

1.  (Optional) Generate synthetic data:
    ```bash
    python generate_data.py
    ```
2.  Run the segmentation pipeline:
    ```bash
    python customer_segmentation.py
    ```

## 📈 Visualizations

### Elbow Curve
Identifies the optimal number of clusters by plotting inertia against the number of clusters.

### Cluster Scatter Plot
Visualizes the customer segments and their centroids in a 2D space.
