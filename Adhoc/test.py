# =============================================================================
# SIMPLE: ONE MODEL PER PPG + PLOT FIT
# =============================================================================

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. LOAD YOUR DATA
# -----------------------------------------------------------------------------
# df = pd.read_csv('your_data.csv')

# Assuming you have df_model ready with these columns:
# ppg_id, ln_base_units, ln_own_price, ln_comp1_price, ln_comp2_price, 
# ln_comp3_price, ln_tdp, ln_promo, trend, sin_52, cos_52, is_holiday
# And also: base_price, base_unit_sales (for plotting)

# -----------------------------------------------------------------------------
# 2. FIT SEPARATE MODEL FOR EACH PPG
# -----------------------------------------------------------------------------

formula = """
ln_base_units ~ 
    ln_own_price 
    + ln_comp1_price 
    + ln_comp2_price 
    + ln_comp3_price
    + ln_tdp 
    + ln_promo
    + trend 
    + sin_52 + cos_52
    + is_holiday
"""

ppg_list = df_model['ppg_id'].unique()
ppg_models = {}  # Store fitted models
ppg_results = [] # Store elasticities

for ppg_id in ppg_list:
    # Filter data for this PPG
    ppg_data = df_model[df_model['ppg_id'] == ppg_id].copy()
    
    # Fit OLS
    model = smf.ols(formula=formula, data=ppg_data).fit()
    ppg_models[ppg_id] = model
    
    # Store results
    ppg_results.append({
        'ppg_id': ppg_id,
        'own_price_elasticity': model.params['ln_own_price'],
        'comp1_elasticity': model.params['ln_comp1_price'],
        'comp2_elasticity': model.params['ln_comp2_price'],
        'comp3_elasticity': model.params['ln_comp3_price'],
        'tdp_elasticity': model.params['ln_tdp'],
        'r_squared': model.rsquared,
        'n_obs': int(model.nobs)
    })

elasticity_df = pd.DataFrame(ppg_results)
print("Elasticities by PPG:")
print(elasticity_df.round(3).to_string(index=False))

# -----------------------------------------------------------------------------
# 3. PLOT MODEL FIT FOR EACH PPG
# -----------------------------------------------------------------------------

ncols = 5
nrows = int(np.ceil(len(ppg_list) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
axes = axes.flatten()

for idx, ppg_id in enumerate(ppg_list):
    ax = axes[idx]
    ppg_data = df_model[df_model['ppg_id'] == ppg_id]
    model = ppg_models[ppg_id]
    
    # Actual scatter
    ax.scatter(ppg_data['base_price'], ppg_data['base_unit_sales'], 
               alpha=0.4, s=20, c='steelblue', label='Actual')
    
    # Fitted curve: predict across price range
    price_range = np.linspace(ppg_data['base_price'].min() * 0.95, 
                               ppg_data['base_price'].max() * 1.05, 50)
    
    # Create prediction dataframe (hold other vars at mean)
    pred_df = pd.DataFrame({
        'ln_own_price': np.log(price_range),
        'ln_comp1_price': ppg_data['ln_comp1_price'].mean(),
        'ln_comp2_price': ppg_data['ln_comp2_price'].mean(),
        'ln_comp3_price': ppg_data['ln_comp3_price'].mean(),
        'ln_tdp': ppg_data['ln_tdp'].mean(),
        'ln_promo': ppg_data['ln_promo'].mean(),
        'trend': ppg_data['trend'].mean(),
        'sin_52': 0,
        'cos_52': 1,
        'is_holiday': 0
    })
    
    fitted_units = np.exp(model.predict(pred_df))
    ax.plot(price_range, fitted_units, color='red', linewidth=2, label='Model Fit')
    
    # Labels
    elas = elasticity_df[elasticity_df['ppg_id'] == ppg_id]['own_price_elasticity'].values[0]
    r2 = elasticity_df[elasticity_df['ppg_id'] == ppg_id]['r_squared'].values[0]
    ax.set_title(f'{ppg_id}\nε={elas:.2f}, R²={r2:.2f}', fontsize=10)
    ax.set_xlabel('Price', fontsize=8)
    ax.set_ylabel('Units', fontsize=8)
    ax.grid(True, alpha=0.3)

# Hide empty subplots
for idx in range(len(ppg_list), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Price vs Demand by PPG (Separate Models)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 4. SINGLE PPG DETAILED PLOT (OPTIONAL)
# -----------------------------------------------------------------------------

def plot_single_ppg(ppg_id):
    """Detailed plot for one PPG."""
    ppg_data = df_model[df_model['ppg_id'] == ppg_id]
    model = ppg_models[ppg_id]
    
    # Price range
    price_range = np.linspace(ppg_data['base_price'].min() * 0.90, 
                               ppg_data['base_price'].max() * 1.10, 50)
    
    # Prediction dataframe
    pred_df = pd.DataFrame({
        'ln_own_price': np.log(price_range),
        'ln_comp1_price': ppg_data['ln_comp1_price'].mean(),
        'ln_comp2_price': ppg_data['ln_comp2_price'].mean(),
        'ln_comp3_price': ppg_data['ln_comp3_price'].mean(),
        'ln_tdp': ppg_data['ln_tdp'].mean(),
        'ln_promo': ppg_data['ln_promo'].mean(),
        'trend': ppg_data['trend'].mean(),
        'sin_52': 0,
        'cos_52': 1,
        'is_holiday': 0
    })
    
    fitted_units = np.exp(model.predict(pred_df))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(ppg_data['base_price'], ppg_data['base_unit_sales'], 
               alpha=0.5, s=50, c='steelblue', label='Actual (Historic)')
    ax.plot(price_range, fitted_units, color='red', linewidth=2.5, label='Model Fit')
    
    elas = model.params['ln_own_price']
    ax.set_title(f'{ppg_id}: Price vs Demand\nOwn-Price Elasticity: {elas:.2f}', fontsize=14)
    ax.set_xlabel('Base Price ($)', fontsize=12)
    ax.set_ylabel('Base Units', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Usage:
# plot_single_ppg('PPG_01')
```

---

## What This Does

| Step | Description |
|------|-------------|
| Loop through PPGs | Fits separate OLS model for each PPG (~100 obs each) |
| Store models | `ppg_models['PPG_01']` gives you that PPG's fitted model |
| Store elasticities | `elasticity_df` has all PPG elasticities in one table |
| Grid plot | Shows all PPGs with scatter + fitted demand curve |
| Single PPG plot | `plot_single_ppg('PPG_01')` for detailed view |

---

## Output Table (Example)
```
ppg_id  own_price_elasticity  comp1_elasticity  r_squared  n_obs
PPG_01                -2.58             0.79       0.92    104
PPG_02                -2.47             0.81       0.89    104
PPG_03                -2.62             0.76       0.91    104
...