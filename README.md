# Simple Linear Regression Model

This project implements a simple linear regression model to analyze and predict relationships between variables, specifically focusing on advertising spend and sales.

## Description

The code demonstrates the complete workflow of building, training, evaluating, and visualizing a linear regression model using Python. The dataset (`business_linear_regression_dataset.csv`) contains business-related data where the goal is to predict sales based on advertising spend.

## Libraries Used

- **pandas**: For loading and handling the dataset.
- **scikit-learn**:
  - `train_test_split`: To split the data into training and testing sets.
  - `LinearRegression`: To create and train the linear regression model.
  - `mean_squared_error`, `r2_score`, `mean_absolute_error`: For model evaluation.
- **matplotlib**: For visualizing the data and regression results.

## Steps in the Code

1. **Import Required Libraries**  
   The code begins by importing the necessary libraries for data processing, modeling, evaluation, and visualization.

2. **Load the Dataset**  
   ```python
   df = pd.read_csv('business_linear_regression_dataset.csv')
   print(df.head())
   ```
   The dataset is loaded into a pandas DataFrame, and the first few rows are printed for inspection.

3. **Define Features and Target Variable**  
   ```python
   x = df.iloc[:, 0:1]  # Independent variable (Advertising Spend)
   y = df.iloc[:, -1]   # Dependent variable (Sales)
   ```
   - `x`: Contains the independent variable(s).
   - `y`: Contains the dependent variable to be predicted.

4. **Split the Dataset**  
   ```python
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
   ```
   The dataset is split into:
   - **Training set (80%)**: Used to train the model.
   - **Testing set (20%)**: Used to evaluate the model's performance.

5. **Train the Model**  
   ```python
   model = LinearRegression()
   model.fit(x_train, y_train)
   ```
   The `LinearRegression` model is instantiated and trained on the training data.

6. **Make Predictions**  
   ```python
   y_pred = model.predict(x_test)
   ```
   The trained model predicts the sales values based on the test data.

7. **Evaluate the Model**  
   ```python
   r2 = r2_score(y_test, y_pred)
   mae = mean_absolute_error(y_test, y_pred)
   mse = mean_squared_error(y_test, y_pred)
   print(f'Mean Absolute Error: {mae}')
   print(f'R-squared: {r2}')
   print(f'Mean Squared Error: {mse}')
   ```
   Key performance metrics are calculated:
   - **R-squared (R²)**: Explains the proportion of variance in the dependent variable explained by the model.
   - **Mean Absolute Error (MAE)**: Measures the average magnitude of errors.
   - **Mean Squared Error (MSE)**: Measures the average squared error.

8. **Visualize the Results**  
   ```python
   plt.scatter(x_test, y_test, color='blue', label='data')
   plt.plot(x_test, y_pred, color='red', label='Regression line')
   plt.xlabel('Advertising_Spend')
   plt.ylabel('Sales')
   plt.title('Linear Regression Model')
   plt.legend()
   plt.show()
   ```
   A scatter plot is generated to display the actual data points, with the regression line overlaid to visualize the model's fit.

## Output

- **Printed Metrics**:  
  - Mean Absolute Error (MAE)
  - R-squared (R²)
  - Mean Squared Error (MSE)
  
- **Visualization**:  
  A scatter plot showing the actual test data points and the fitted regression line.

## How to Run the Code

1. Ensure you have the following libraries installed: `pandas`, `scikit-learn`, and `matplotlib`.
2. Save the dataset as `business_linear_regression_dataset.csv` in the working directory.
3. Run the script to see the evaluation metrics and the visualization.

