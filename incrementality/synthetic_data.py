import pandas as pd
import numpy as np
from scipy.special import softmax

# Set seed for reproducibility
np.random.seed(42)

def generate_synthetic_cpg_data(n_weeks=104):
    # --- 1. Define Product Hierarchy ---
    products = [
        # Nest: Soda | Manufacturer: BigFizz Co
        {'sku': 'SKU_01', 'brand': 'Brand_A', 'manuf': 'BigFizz', 'nest': 'Soda', 'base_price': 1.50, 'base_pref': 2.0},
        {'sku': 'SKU_02', 'brand': 'Brand_A', 'manuf': 'BigFizz', 'nest': 'Soda', 'base_price': 2.50, 'base_pref': 1.8}, # Larger size
        {'sku': 'SKU_03', 'brand': 'Brand_B', 'manuf': 'BigFizz', 'nest': 'Soda', 'base_price': 1.40, 'base_pref': 1.5},
        
        # Nest: Soda | Manufacturer: PopCorp
        {'sku': 'SKU_04', 'brand': 'Brand_C', 'manuf': 'PopCorp', 'nest': 'Soda', 'base_price': 1.60, 'base_pref': 1.9},
        {'sku': 'SKU_05', 'brand': 'Brand_C', 'manuf': 'PopCorp', 'nest': 'Soda', 'base_price': 1.60, 'base_pref': 1.9}, # Flavor variant
        
        # Nest: Juice | Manufacturer: NatureDrink
        {'sku': 'SKU_06', 'brand': 'Brand_D', 'manuf': 'NatureDrink', 'nest': 'Juice', 'base_price': 3.00, 'base_pref': 1.2},
        {'sku': 'SKU_07', 'brand': 'Brand_E', 'manuf': 'NatureDrink', 'nest': 'Juice', 'base_price': 3.50, 'base_pref': 1.0},
    ]
    
    dates = pd.date_range(start='2023-01-01', periods=n_weeks, freq='W-MON')
    
    all_rows = []

    # Market Sensitivity Parameters
    price_elasticity = -2.5  # People are price sensitive
    nest_correlation = 0.5   # Correlation of error terms within nests (for Phase 2)
    
    # --- 2. Generate Time Series Data ---
    for date in dates:
        # Market Size (Total potential units sold in category that week)
        # Add seasonality: Peaks in summer (approx week 26)
        week_num = date.weekofyear
        seasonality = 1 + 0.3 * np.sin(2 * np.pi * week_num / 52)
        market_size = 50000 * seasonality 
        
        # Nest-specific random shock (for Nested Logit validity)
        soda_shock = np.random.normal(0, 0.2)
        juice_shock = np.random.normal(0, 0.2)
        
        week_data = []
        utilities = []
        
        for p in products:
            # A. Simulate Predictors
            # Price fluctuates randomly around base price
            price_shock = np.random.uniform(-0.10, 0.10)
            current_price = round(p['base_price'] * (1 + price_shock), 2)
            
            # ACV (Availability): Varies between 0.50 and 1.00
            acv = round(np.random.uniform(0.50, 1.0), 2)
            
            # B. Simulate Utility (The "Choice" Signal)
            # Utility = Base + (Price * Elasticity) + Nest_Shock + Random_Noise
            nest_shock = soda_shock if p['nest'] == 'Soda' else juice_shock
            
            # Note: ACV affects the FINAL sales, not necessarily the intrinsic utility,
            # but in aggregate data, low ACV acts like a penalty to utility/probability.
            # We will use it as a multiplier for sales later.
            
            utility = (p['base_pref'] + 
                       (current_price * price_elasticity) + 
                       nest_shock + 
                       np.random.normal(0, 0.1))
            
            week_data.append({
                'Date': date,
                'SKU': p['sku'],
                'Brand': p['brand'],
                'Manufacturer': p['manuf'],
                'Nest': p['nest'], # Helpful for Phase 2
                'Price': current_price,
                'ACV': acv,
                'Utility_Hidden': utility # We won't export this, but we need it for calc
            })
            utilities.append(utility)
            
        # C. Calculate Shares (Softmax)
        # In a discrete choice, probability = exp(u) / sum(exp(u))
        probs = softmax(utilities)
        
        # D. Assign Volumes
        for i, item in enumerate(week_data):
            # Sales = Market Size * Probability * ACV * Noise
            # ACV here acts as a physical constraint (can't buy if not there)
            expected_sales = market_size * probs[i] * item['ACV']
            
            # Add Poisson noise for integer sales counts
            actual_sales = np.random.poisson(expected_sales)
            
            item['Sales_Volume'] = actual_sales
            all_rows.append(item)

    df = pd.DataFrame(all_rows)

    # --- 3. Create "Unbalanced" Structure ---
    # Randomly drop 15% of rows to simulate products not being listed/tracked in certain weeks
    mask = np.random.rand(len(df)) > 0.15
    df_unbalanced = df[mask].copy()
    
    # Drop the hidden utility column (since you wouldn't have this in real life)
    df_final = df_unbalanced.drop(columns=['Utility_Hidden'])
    
    return df_final

# Generate the data
df = generate_synthetic_cpg_data()
df.to_csv("sales_data.csv")
# Display snippet
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

print("\nExample of Unbalanced Structure (SKU Counts per week):")
print(df.groupby('Date')['SKU'].count().head())