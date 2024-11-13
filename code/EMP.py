import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA


import numpy as np
import pandas as pd
from scipy.optimize import minimize

#####################Empirical Distribution and Likelihood Example##################################
# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for 200 patients
n_samples = 200
data = {
    'Age': np.random.randint(18, 90, size=n_samples),
    'BMI': np.random.normal(25, 5, size=n_samples),
    'Num_Medications': np.random.poisson(5, size=n_samples),
    'Num_Visits': np.random.poisson(2, size=n_samples),
    'Has_Complications': np.random.binomial(1, 0.3, size=n_samples),
    'Readmission': np.random.binomial(1, 0.4, size=n_samples)
}

df = pd.DataFrame(data)
print(df.head())

# Define subset of data where Readmission == 1
readmitted = df[df['Readmission'] == 1]

# Target: estimating mean and variance of BMI for readmitted patients using empirical likelihood

# Define the empirical likelihood function with both mean and variance constraints
def empirical_likelihood(params, data):
    # `params` includes the weight vector `p` and the mean estimate `theta` and variance estimate `sigma2`
    theta = params[0]  # Mean estimate of BMI
    sigma2 = params[1]  # Variance estimate of BMI
    p = params[2:]  # Weights for each observation
    
    # Ensure weights sum to 1 (normalization)
    p = np.exp(p)  # Exponential to ensure positivity
    p /= np.sum(p)
    
    # Moment conditions: The weighted mean should equal theta, and the weighted variance should equal sigma2
    weighted_mean = np.sum(p * data['BMI'])
    mean_condition = weighted_mean - theta
    
    # Compute the weighted variance condition
    weighted_variance = np.sum(p * (data['BMI'] - theta)**2)
    variance_condition = weighted_variance - sigma2
    
    # Log-likelihood based on weights (penalizing deviation from equal weights)
    log_likelihood = np.sum(np.log(p))
    
    # Objective: maximize log likelihood while enforcing moment conditions
    penalty = 1e6 * (mean_condition**2 + variance_condition**2)  # Penalty term for both conditions
    return -(log_likelihood - penalty)  # Negative for minimization

# Initial guesses
initial_theta = readmitted['BMI'].mean()  # Start with the sample mean
initial_sigma2 = readmitted['BMI'].var()  # Start with the sample variance
initial_weights = np.zeros(len(readmitted))  # Initial weights log-transformed

# Combine initial parameters
initial_params = np.concatenate(([initial_theta, initial_sigma2], initial_weights))

# Optimization using Nelder-Mead (no gradient needed)
result = minimize(empirical_likelihood, initial_params, args=(readmitted,),
                  method='Nelder-Mead', options={'disp': True})

# Extract optimized parameters
theta_estimate = result.x[0]
sigma2_estimate = result.x[1]
weights = np.exp(result.x[2:]) / np.sum(np.exp(result.x[2:]))

print(f"Estimated Mean BMI (Empirical Likelihood): {theta_estimate:.2f}")
print(f"Estimated Variance BMI (Empirical Likelihood): {sigma2_estimate:.2f}")
print(f"Weights: {weights}")


#1) ______________________________________________________________
######synthetic_data: EMP likelihood method############

# Seed for reproducibility
np.random.seed(42)

# 1) Data Generation: Generate synthetic time series data with ARMA(1,1), seasonal component, and outliers
def generate_synthetic_data(n=120, p=1, q=1, seasonal_period=12, seasonal_harmonics=2, outlier_freq=10):
    phi = [0.5]  # AR coefficient
    theta = [0.3]  # MA coefficient
    alpha = [0.1] * seasonal_harmonics
    beta = [0.2] * seasonal_harmonics
    y = np.zeros(n)
    epsilon = np.random.normal(0, 1, n)

    for t in range(max(p, q), n):
        arma_term = sum(phi[i] * y[t - i - 1] for i in range(p)) + sum(theta[j] * epsilon[t - j - 1] for j in range(q))
        seasonality = sum(
            alpha[k] * np.sin(2 * np.pi * (k + 1) * t / seasonal_period) +
            beta[k] * np.cos(2 * np.pi * (k + 1) * t / seasonal_period)
            for k in range(seasonal_harmonics)
        )
        y[t] = arma_term + seasonality + epsilon[t]

    outliers = np.arange(0, n, outlier_freq)
    y[outliers] += np.random.normal(10, 5, size=len(outliers))  # Larger random deviations as outliers
    return y

# Generate data
y = generate_synthetic_data()

# Split data into training and testing sets
train_size = int(len(y) * 0.8)
train, test = y[:train_size], y[train_size:]

# 2) Define Empirical Likelihood Functions with Improved Log Stability
def empirical_likelihood_loss(params, y, p, q, seasonal_period=None, seasonal_harmonics=0, epsilon_small=1e-6):
    n = len(y)
    phi = params[:p]  # AR coefficients
    theta = params[p:p + q]  # MA coefficients
    lambda_val = params[p + q]  # Lagrange multiplier for empirical likelihood

    alpha = params[p + q + 1:p + q + 1 + seasonal_harmonics]
    beta = params[p + q + 1 + seasonal_harmonics:p + q + 1 + 2 * seasonal_harmonics]

    epsilon = np.zeros(n)
    for t in range(max(p, q), n):
        arma_term = sum(phi[i] * y[t - i - 1] for i in range(p)) + sum(theta[j] * epsilon[t - j - 1] for j in range(q))
        seasonality = sum(
            alpha[k] * np.sin(2 * np.pi * (k + 1) * t / seasonal_period) +
            beta[k] * np.cos(2 * np.pi * (k + 1) * t / seasonal_period)
            for k in range(seasonal_harmonics)
        )
        epsilon[t] = y[t] - arma_term - seasonality

    # Adding a small constant to stabilize the log term
    adjusted_residuals = 1 + lambda_val * epsilon[max(p, q):]
    loss = np.sum(np.log(epsilon_small + adjusted_residuals))
    return loss

def fit_arma_empirical_likelihood(y, p=1, q=1, seasonal_period=None, seasonal_harmonics=0, initial_params=None):
    if initial_params is None:
        initial_params = np.random.rand(p + q + 1 + 2 * seasonal_harmonics) * 0.1  # Smaller random start
    bounds = [(None, None)] * (p + q) + [(0, 10)] + [(None, None)] * (2 * seasonal_harmonics)
    result = minimize(
        empirical_likelihood_loss, initial_params,
        args=(y, p, q, seasonal_period, seasonal_harmonics),
        method='Nelder-Mead', bounds=bounds
    )
    params = result.x
    return params

def forecast(y, params, p, q, steps=10, seasonal_period=None, seasonal_harmonics=0):
    phi = params[:p]
    theta = params[p:p + q]
    alpha = params[p + q + 1:p + q + 1 + seasonal_harmonics]
    beta = params[p + q + 1 + seasonal_harmonics:p + q + 1 + 2 * seasonal_harmonics]

    epsilon = np.zeros(len(y))
    forecast_values = []
    for t in range(steps):
        arma_term = sum(phi[i] * y[-i - 1] for i in range(p)) + sum(theta[j] * epsilon[-j - 1] for j in range(q))
        seasonality = sum(
            alpha[k] * np.sin(2 * np.pi * (k + 1) * len(y) / seasonal_period) +
            beta[k] * np.cos(2 * np.pi * (k + 1) * len(y) / seasonal_period)
            for k in range(seasonal_harmonics)
        )
        y_forecast = arma_term + seasonality
        forecast_values.append(y_forecast)
        y = np.append(y, y_forecast)
        epsilon = np.append(epsilon, 0)
    return forecast_values

# 3) Fit Empirical Likelihood Model and Forecast
params_el = fit_arma_empirical_likelihood(train, p=1, q=1, seasonal_period=12, seasonal_harmonics=2)
forecast_el = forecast(train, params_el, p=1, q=1, steps=len(test), seasonal_period=12, seasonal_harmonics=2)

# 4) Fit Standard ARMA Model and Forecast for Comparison
arma_model = ARIMA(train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0))
arma_fit = arma_model.fit()
forecast_arma = arma_fit.forecast(steps=len(test))

# 5) Evaluate Performance of Both Models
def evaluate_forecast(actual, forecast):
    mape = mean_absolute_percentage_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae = mean_absolute_error(actual, forecast)
    return {'MAPE': mape, 'RMSE': rmse, 'MAE': mae}

# Evaluation
evaluation_el = evaluate_forecast(test, forecast_el)
evaluation_arma = evaluate_forecast(test, forecast_arma)

print("Empirical Likelihood ARMA Forecast Performance:")
print(f"MAPE: {evaluation_el['MAPE']:.4f}, RMSE: {evaluation_el['RMSE']:.4f}, MAE: {evaluation_el['MAE']:.4f}")

print("\nStandard ARMA Forecast Performance:")
print(f"MAPE: {evaluation_arma['MAPE']:.4f}, RMSE: {evaluation_arma['RMSE']:.4f}, MAE: {evaluation_arma['MAE']:.4f}")


#2)________________________________________________________________
######################electricity data##########################
#####apply ARMA on transformation EMP without EMP likelihood#########

##########several back testing with EMP transformation only##############
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['DATE'] = pd.to_datetime(data['DATE'])  # Convert DATE to datetime format
    data = data.set_index('DATE')  # Set DATE as an index
    return data['IPG2211A2N']

# Empirical transformation and inverse transformation functions
def empirical_transform(data, normalize=True):
    ranks = rankdata(data, method='average')
    empirical_data = ranks / (len(data) + 1) if normalize else ranks
    return empirical_data

def inverse_empirical_transform(empirical_data, original_data):
    sorted_original = np.sort(original_data)
    n = len(sorted_original)
    indices = (empirical_data * (n - 1)).astype(int)  # Map empirical values back to original space
    indices = np.clip(indices, 0, n - 1)  # Ensure indices stay within bounds
    return sorted_original[indices]

# Function to train and forecast using Empirical Transformed ARMA
def empirical_arma_forecast(train_data, test_data, order=(1, 0, 1), forecast_steps=5, normalize=True):
    # Transform training data to empirical space
    train_empirical = empirical_transform(train_data, normalize=normalize)
    
    # Fit ARMA model on empirical data
    arma_empirical_model = ARIMA(train_empirical, order=order)
    arma_empirical_fit = arma_empirical_model.fit()
    
    # Forecast in empirical space and inverse transform to original space
    forecast_empirical = arma_empirical_fit.forecast(steps=len(test_data))
    forecast_original = inverse_empirical_transform(forecast_empirical, train_data)
    
    # Calculate moving average for forecast (default 5 months)
    forecast_moving_avg = pd.Series(forecast_original, index=test_data.index).rolling(window=forecast_steps).mean().dropna()
    return forecast_moving_avg

# Function to train and forecast using Standard ARMA
def standard_arma_forecast(train_data, test_data, order=(1, 0, 1), forecast_steps=5):
    # Fit standard ARMA model
    arma_standard_model = ARIMA(train_data, order=order)
    arma_standard_fit = arma_standard_model.fit()
    
    # Forecast and calculate moving average
    forecast_standard = arma_standard_fit.forecast(steps=len(test_data))
    forecast_standard_moving_avg = pd.Series(forecast_standard, index=test_data.index).rolling(window=forecast_steps).mean().dropna()
    return forecast_standard_moving_avg

# Evaluation function for forecasted data
def evaluate_forecast(actual, forecast):
    mape = mean_absolute_percentage_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae = mean_absolute_error(actual, forecast)
    return {'MAPE': mape, 'RMSE': rmse, 'MAE': mae}

# Backtesting function with multiple training sizes
def backtest_empirical_and_standard_arma(file_path, order=(1, 0, 1), forecast_steps=5, normalize=True, test_size=5, backtest_points=[5, 10, 20, 50]):
    # Load data
    series = load_data(file_path)
    n = len(series)
    
    # Store performance metrics for each backtest
    results = {"Empirical Transformed ARMA": [], "Standard ARMA": []}
    
    # Perform backtesting
    for p in backtest_points:
        train_end = n - p - test_size
        train_data = series[:train_end]
        test_data = series[train_end:train_end + test_size]
        
        # Check if training and test data have sufficient data points
        if len(train_data) == 0 or len(test_data) < forecast_steps:
            print(f"Skipping backtest with p={p} due to insufficient data.")
            continue
        
        # Forecast using Empirical Transformed ARMA and Standard ARMA
        forecast_empirical = empirical_arma_forecast(train_data, test_data, order=order, forecast_steps=forecast_steps, normalize=normalize)
        forecast_standard = standard_arma_forecast(train_data, test_data, order=order, forecast_steps=forecast_steps)
        
        # Check if forecast results are not empty
        if len(forecast_empirical) == 0 or len(forecast_standard) == 0:
            print(f"Skipping backtest with p={p} due to empty forecast results.")
            continue
        
        # Align test data to match the forecast length
        test_moving_avg = test_data.rolling(window=forecast_steps).mean().dropna()
        if len(test_moving_avg) == 0:
            print(f"Skipping backtest with p={p} due to insufficient test data for rolling average.")
            continue
        test_aligned = test_moving_avg.iloc[:len(forecast_empirical)]
        
        # Evaluate forecasts
        evaluation_empirical = evaluate_forecast(test_aligned, forecast_empirical)
        evaluation_standard = evaluate_forecast(test_aligned, forecast_standard)
        
        # Store results with model type and backtest case
        results["Empirical Transformed ARMA"].append({
            'case': f'Backtest with p={p}',
            'MAPE': evaluation_empirical['MAPE'],
            'RMSE': evaluation_empirical['RMSE'],
            'MAE': evaluation_empirical['MAE']
        })
        
        results["Standard ARMA"].append({
            'case': f'Backtest with p={p}',
            'MAPE': evaluation_standard['MAPE'],
            'RMSE': evaluation_standard['RMSE'],
            'MAE': evaluation_standard['MAE']
        })
        
        # Print results for each backtest
        print(f"\nBacktest with Training End at t={train_end}")
        print("Empirical Transformed ARMA Forecast Performance:")
        print(f"MAPE: {evaluation_empirical['MAPE']:.4f}, RMSE: {evaluation_empirical['RMSE']:.4f}, MAE: {evaluation_empirical['MAE']:.4f}")
        
        print("Standard ARMA Forecast Performance:")
        print(f"MAPE: {evaluation_standard['MAPE']:.4f}, RMSE: {evaluation_standard['RMSE']:.4f}, MAE: {evaluation_standard['MAE']:.4f}")
    
    return results

# Example usage
file_path = 'Electric_Production_tm.csv'
results = backtest_empirical_and_standard_arma(file_path, order=(1, 0, 1), forecast_steps=3, normalize=True, test_size=3, backtest_points=[10, 30, 50, 75, 100])

# Print final structured results
print("\nFinal Results:\n", results)

results = backtest_empirical_and_standard_arma(file_path, order=(1, 0, 1), forecast_steps=3, normalize=True, test_size=3, backtest_points=[10, 30, 50, 75, 100])



#3)________________________________________________________________

#######EMP likelihood on log transform raw and transfor back##############
###########################################################################

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['DATE'] = pd.to_datetime(data['DATE'])  # Convert DATE to datetime format
    data = data.set_index('DATE')  # Set DATE as an index
    return data['IPG2211A2N']

# Empirical likelihood loss function with regularization
def empirical_likelihood_loss(params, y, p, q, epsilon_small=1e-6, regularization=1e-1):
    n = len(y)
    phi = params[:p]  # AR coefficients
    theta = params[p:p + q]  # MA coefficients
    lambda_val = params[p + q]  # Lagrange multiplier for empirical likelihood

    epsilon = np.zeros(n)
    for t in range(max(p, q), n):
        arma_term = sum(phi[i] * y[t - i - 1] for i in range(p)) + sum(theta[j] * epsilon[t - j - 1] for j in range(q))
        epsilon[t] = y[t] - arma_term

    adjusted_residuals = 1 + lambda_val * epsilon[max(p, q):]
    loss = np.sum(np.log(epsilon_small + adjusted_residuals)) + regularization * np.sum(params**2)
    return loss

# Fit ARMA model with empirical likelihood on log-transformed data
def fit_arma_empirical_likelihood_log_data(train_log, p=1, q=1):
    initial_params = np.concatenate([np.zeros(p + q), [0.01]])  # AR, MA, and lambda multiplier
    bounds = [(-2, 2)] * (p + q) + [(1e-4, 10)]  # Bounds for AR, MA coefficients and lambda

    result = minimize(
        empirical_likelihood_loss, initial_params,
        args=(train_log, p, q),
        method='Powell',
        bounds=bounds,
        options={'xtol': 1e-3, 'ftol': 1e-3, 'maxiter': 1000}
    )
    return result.x if result.success else None

# Forecast function using fitted parameters on log-transformed data
def forecast_log_data(y, params, p, q, steps=10):
    phi = params[:p]
    theta = params[p:p + q]
    epsilon = np.zeros(len(y))
    forecast_values = []
    for t in range(steps):
        arma_term = sum(phi[i] * y[-i - 1] for i in range(p)) + sum(theta[j] * epsilon[-j - 1] for j in range(q))
        y_forecast = arma_term
        forecast_values.append(y_forecast)
        y = np.append(y, y_forecast)
        epsilon = np.append(epsilon, 0)
    return np.array(forecast_values)

# Function to evaluate the model
def evaluate_forecast(actual, forecast):
    mape = mean_absolute_percentage_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae = mean_absolute_error(actual, forecast)
    return {'MAPE': mape, 'RMSE': rmse, 'MAE': mae}

# Main backtesting function for EMP likelihood ARMA on log-transformed data
def backtest_empirical_likelihood_log_data(file_path, p=1, q=1, forecast_steps=5, test_size=3, backtest_points=[10, 30, 50, 75, 100]):
    series = load_data(file_path)
    n = len(series)
    results = {"Empirical Likelihood ARMA": [], "Standard ARMA": []}

    # Perform backtesting
    for bp in backtest_points:
        train_end = n - bp - test_size
        train_data = series[:train_end]
        test_data = series[train_end:train_end + test_size]

        # Log-transform the training and test data
        train_log = np.log(train_data)
        test_log = np.log(test_data)

        # Fit Empirical Likelihood model on log-transformed data
        params_el_log_data = fit_arma_empirical_likelihood_log_data(train_log, p=p, q=q)
        if params_el_log_data is None:
            print(f"Skipping backtest with p={bp} due to optimization failure.")
            continue

        # Forecast using the empirical likelihood model on log-transformed data
        forecast_emp_log = forecast_log_data(train_log.values, params_el_log_data, p=p, q=q, steps=test_size)
        forecast_emp_original = np.exp(forecast_emp_log)  # Transform back to the original scale
        forecast_emp_moving_avg = pd.Series(forecast_emp_original, index=test_data.index).rolling(window=forecast_steps).mean().dropna()

        # Fit Standard ARMA Model on log-transformed data for comparison
        arma_model = ARIMA(train_log, order=(p, 0, q))
        arma_fit = arma_model.fit()
        forecast_arma_log = arma_fit.forecast(steps=test_size)
        forecast_arma_original = np.exp(forecast_arma_log)  # Transform back to original scale
        forecast_arma_moving_avg = pd.Series(forecast_arma_original, index=test_data.index).rolling(window=forecast_steps).mean().dropna()

        # Calculate moving average of the actual test data on the original scale
        test_moving_avg = test_data.rolling(window=forecast_steps).mean().dropna()
        test_aligned = test_moving_avg.iloc[:len(forecast_emp_moving_avg)]

        # Evaluate forecasts
        evaluation_empirical = evaluate_forecast(test_aligned, forecast_emp_moving_avg)
        evaluation_standard = evaluate_forecast(test_aligned, forecast_arma_moving_avg)

        # Store results with model type and backtest case
        results["Empirical Likelihood ARMA"].append({
            'case': f'Backtest with p={bp}',
            'MAPE': evaluation_empirical['MAPE'],
            'RMSE': evaluation_empirical['RMSE'],
            'MAE': evaluation_empirical['MAE']
        })
        
        results["Standard ARMA"].append({
            'case': f'Backtest with p={bp}',
            'MAPE': evaluation_standard['MAPE'],
            'RMSE': evaluation_standard['RMSE'],
            'MAE': evaluation_standard['MAE']
        })

        # Print results for each backtest
        print(f"\nBacktest with Training End at t={train_end}")
        print("Empirical Likelihood ARMA on Log-Transformed Data Forecast Performance:")
        print(f"MAPE: {evaluation_empirical['MAPE']:.4f}, RMSE: {evaluation_empirical['RMSE']:.4f}, MAE: {evaluation_empirical['MAE']:.4f}")
        
        print("Standard ARMA on Log-Transformed Data Forecast Performance:")
        print(f"MAPE: {evaluation_standard['MAPE']:.4f}, RMSE: {evaluation_standard['RMSE']:.4f}, MAE: {evaluation_standard['MAE']:.4f}")

    return results

# Example usage
file_path = 'Electric_Production_tm.csv'
results = backtest_empirical_likelihood_log_data(file_path, p=1, q=1, forecast_steps=5, test_size=5, backtest_points=[10, 40, 70, 100, 130])
results = backtest_empirical_likelihood_log_data(file_path, p=1, q=1, forecast_steps=3, test_size=3, backtest_points=[10, 40, 70, 100, 130])

# Print final structured results
print("\nFinal Results:\n", results)
