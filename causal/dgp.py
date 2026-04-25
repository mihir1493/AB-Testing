import numpy as np
import pandas as pd

# --- Parameters for Data Generation ---
N_WEEKS = 104  # 2 years of weekly data
BASE_SALES = 5000
BASE_PRICE_A = 12.00
BASE_PRICE_B = 11.50
BASE_PRICE_C = 12.50

# True Elasticities (these are the parameters PyMC will try to estimate)
# Negative own-price elasticity, positive cross-price elasticities (substitutes)
TRUE_ELASTICITY_A = -3.0
TRUE_ELASTICITY_B = 1.5
TRUE_ELASTICITY_C = 1.0

# Seasonality Parameters (simple annual cycle)
SEASONALITY_AMPLITUDE = 0.05  # 5% variation
SEASONALITY_PERIOD = 52.0  # Weeks in a year

# --- Create DataFrame ---
df = pd.DataFrame({'Week': np.arange(1, N_WEEKS + 1)})
df['Date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(df['Week'] * 7, unit='days')

# --- 1. Generate Prices ---
# Prices are base with a small, random weekly variation (e.g., +/- 10%)
np.random.seed(42)
df['Price_A'] = np.clip(np.random.normal(BASE_PRICE_A, 0.5, N_WEEKS), 10.0, 14.0).round(2)
df['Price_B'] = np.clip(np.random.normal(BASE_PRICE_B, 0.4, N_WEEKS), 9.5, 13.0).round(2)
df['Price_C'] = np.clip(np.random.normal(BASE_PRICE_C, 0.6, N_WEEKS), 11.0, 15.0).round(2)

# --- 2. Calculate Log Sales based on Elasticity Model ---
# Log(Sales) = Intercept + E_A*Log(P_A) + E_B*Log(P_B) + E_C*Log(P_C) + Seasonality + Noise

# Calculate the required intercept to center sales around BASE_SALES
log_base_sales = np.log(BASE_SALES)
log_base_price_A = np.log(BASE_PRICE_A)
log_base_price_B = np.log(BASE_PRICE_B)
log_base_price_C = np.log(BASE_PRICE_C)

# Intercept = Log(Sales) - Sum(E * Log(P))
TRUE_INTERCEPT = (
    log_base_sales - 
    (TRUE_ELASTICITY_A * log_base_price_A) -
    (TRUE_ELASTICITY_B * log_base_price_B) -
    (TRUE_ELASTICITY_C * log_base_price_C)
)

# Apply the model
log_sales_pred = (
    TRUE_INTERCEPT +
    TRUE_ELASTICITY_A * np.log(df['Price_A']) +
    TRUE_ELASTICITY_B * np.log(df['Price_B']) +
    TRUE_ELASTICITY_C * np.log(df['Price_C'])
)

# --- 3. Add Seasonality and Noise ---
# Simple sine function for annual seasonality
seasonality = SEASONALITY_AMPLITUDE * np.sin(2 * np.pi * df['Week'] / SEASONALITY_PERIOD)

# Add random noise (Residuals)
noise = np.random.normal(0, 0.1, N_WEEKS)

log_sales_a = log_sales_pred + seasonality + noise

# --- 4. Convert Log Sales back to Unit Sales ---
df['Sales_A'] = np.exp(log_sales_a).round(0).astype(int)

# Select and display the final columns
df_final = df[['Date', 'Week', 'Price_A', 'Price_B', 'Price_C', 'Sales_A']]

df_final.to_csv('data.csv', index=False)