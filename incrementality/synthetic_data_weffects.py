import pandas as pd
import numpy as np
from scipy.special import softmax

np.random.seed(123)

def generate_cannibalization_data(n_weeks=104):
    # --- 1. Define Product Hierarchy ---
    products = [
        {'sku': 'SKU_01', 'brand': 'Brand_A', 'nest': 'Soda', 'base_price': 1.50, 'base_pref': 2.0},
        {'sku': 'SKU_02', 'brand': 'Brand_A', 'nest': 'Soda', 'base_price': 2.50, 'base_pref': 1.8},
        {'sku': 'SKU_03', 'brand': 'Brand_B', 'nest': 'Soda', 'base_price': 1.40, 'base_pref': 1.5},
        {'sku': 'SKU_04', 'brand': 'Brand_C', 'nest': 'Soda', 'base_price': 1.60, 'base_pref': 1.9},
        {'sku': 'SKU_06', 'brand': 'Brand_D', 'nest': 'Juice', 'base_price': 3.00, 'base_pref': 1.2},
        {'sku': 'SKU_07', 'brand': 'Brand_E', 'nest': 'Juice', 'base_price': 3.50, 'base_pref': 1.0},
    ]
    
    dates = pd.date_range(start='2023-01-01', periods=n_weeks, freq='W-MON')
    all_rows = []

    # --- 2. Iterate Through Time ---
    for date in dates:
        # Market Size (Seasonality)
        week_num = date.weekofyear
        seasonality = 1 + 0.3 * np.sin(2 * np.pi * week_num / 52)
        market_size = 50000 * seasonality 
        
        # --- CRITICAL STEP: Determine Availability (The Choice Set) ---
        # Randomly decide which products are actually on the shelf this week.
        # Let's say there is a 15% chance a product is NOT available (Stockout/Delisted)
        available_products = []
        for p in products:
            if np.random.rand() > 0.15: 
                # Create a copy so we don't mutate the master list
                p_week = p.copy()
                
                # Dynamic Predictors
                p_week['Price'] = round(p['base_price'] * (1 + np.random.uniform(-0.1, 0.1)), 2)
                p_week['ACV'] = round(np.random.uniform(0.6, 1.0), 2)
                p_week['Date'] = date
                
                # Calculate Utility Component
                # Utility = Base + (Price * Elasticity) + Noise
                # Note: We are ignoring the 'Nest Shock' for simplicity in Phase 1, 
                # but you would add it here for Phase 2.
                price_elasticity = -2.5
                p_week['Utility'] = (p['base_pref'] + 
                                     (p_week['Price'] * price_elasticity) + 
                                     np.random.normal(0, 0.1))
                
                available_products.append(p_week)
        
        # If no products are available (rare), skip the week
        if not available_products:
            continue

        # --- 3. Calculate Shares on the RESTRICTED Set ---
        # This is where cannibalization happens. 
        # Because we only feed the 'available_products' into softmax, 
        # the probabilities sum to 100% across ONLY these items.
        utilities = [p['Utility'] for p in available_products]
        probs = softmax(utilities)
        
        # --- 4. Assign Sales ---
        for idx, item in enumerate(available_products):
            # Sales = Market Size * Share * ACV
            # (ACV here acts as a distribution constraint, reducing realization of that share)
            share = probs[idx]
            expected_sales = market_size * share * item['ACV']
            
            item['Sales_Volume'] = np.random.poisson(expected_sales)
            
            # Remove the raw utility before saving to dataframe (since we don't observe it)
            del item['Utility'] 
            del item['base_pref']
            del item['base_price']
            
            all_rows.append(item)

    df_final = pd.DataFrame(all_rows)
    return df_final

df = generate_cannibalization_data()
df.to_csv('sales_data2.csv')
# --- Verification ---
# Let's check if the mean sales of SKU_01 are higher when SKU_03 is missing.
# sku1_when_sku3_present = df[df['Date'].isin(df[df['SKU'] == 'SKU_03']['Date'])]
# sku1_when_sku3_missing = df[~df['Date'].isin(df[df['SKU'] == 'SKU_03']['Date'])]

# avg_sales_present = sku1_when_sku3_present[sku1_when_sku3_present['SKU'] == 'SKU_01']['Sales_Volume'].mean()
# avg_sales_missing = sku1_when_sku3_missing[sku1_when_sku3_missing['SKU'] == 'SKU_01']['Sales_Volume'].mean()

# print(f"SKU_01 Avg Sales when SKU_03 is ON SHELF: {avg_sales_present:.0f}")
# print(f"SKU_01 Avg Sales when SKU_03 is MISSING:  {avg_sales_missing:.0f}")
# print(f"Shift Detected: {((avg_sales_missing - avg_sales_present) / avg_sales_present)*100:.1f}% increase")