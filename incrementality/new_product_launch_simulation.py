import pandas as pd
import numpy as np
from scipy.special import softmax

np.random.seed(42)

def generate_new_product_launch_data(n_weeks_before=52, n_weeks_after=52):
    """
    Simulate a new product launch scenario with cannibalization and incrementality effects.

    Market Structure:
    - Brand A (Your brand): SKU_A1, SKU_A2, SKU_A3 (NEW - launches after n_weeks_before)
    - Brand B (Competitor 1): SKU_B1, SKU_B2
    - Brand C (Competitor 2): SKU_C1

    When SKU_A3 launches:
    - 40% of its sales come from cannibalization of SKU_A1 and SKU_A2
    - 60% of its sales are incremental (taken from Brand B and C)
    """

    # --- Product Definitions ---
    products = [
        # BRAND A (Your Brand) - Existing products
        {'sku': 'SKU_A1', 'brand': 'Brand_A', 'base_price': 2.00, 'base_pref': 2.2, 'launch_week': 0},
        {'sku': 'SKU_A2', 'brand': 'Brand_A', 'base_price': 2.50, 'base_pref': 2.0, 'launch_week': 0},

        # BRAND B (Competitor 1)
        {'sku': 'SKU_B1', 'brand': 'Brand_B', 'base_price': 1.90, 'base_pref': 1.9, 'launch_week': 0},
        {'sku': 'SKU_B2', 'brand': 'Brand_B', 'base_price': 2.20, 'base_pref': 1.8, 'launch_week': 0},

        # BRAND C (Competitor 2)
        {'sku': 'SKU_C1', 'brand': 'Brand_C', 'base_price': 2.10, 'base_pref': 1.7, 'launch_week': 0},

        # BRAND A - NEW PRODUCT (launches at week n_weeks_before)
        {'sku': 'SKU_A3', 'brand': 'Brand_A', 'base_price': 2.30, 'base_pref': 2.5, 'launch_week': n_weeks_before},
    ]

    total_weeks = n_weeks_before + n_weeks_after
    dates = pd.date_range(start='2023-01-01', periods=total_weeks, freq='W-MON')

    all_rows = []

    # Market parameters
    price_elasticity = -2.0

    # --- Generate Time Series Data ---
    for week_idx, date in enumerate(dates):
        # Market size with seasonality
        week_num = date.isocalendar()[1]  # ISO week number
        seasonality = 1 + 0.2 * np.sin(2 * np.pi * week_num / 52)
        market_size = 100000 * seasonality

        # Determine which products are available this week
        available_products = []
        utilities = []

        for p in products:
            # Skip products that haven't launched yet
            if week_idx < p['launch_week']:
                continue

            # Simulate predictors
            price_shock = np.random.uniform(-0.08, 0.08)
            current_price = round(p['base_price'] * (1 + price_shock), 2)

            # ACV: New product ramps up availability, existing products have stable ACV
            if p['sku'] == 'SKU_A3' and week_idx < p['launch_week'] + 12:
                # New product ramps up over 12 weeks
                weeks_since_launch = week_idx - p['launch_week']
                acv = round(min(0.4 + (weeks_since_launch * 0.05), 0.95), 2)
            else:
                acv = round(np.random.uniform(0.80, 0.98), 2)

            # Calculate utility
            # Base preference + price effect + random noise
            utility = (p['base_pref'] +
                      (current_price * price_elasticity) +
                      np.random.normal(0, 0.15))

            # --- KEY: Model Cannibalization Effect ---
            # When SKU_A3 is present, reduce utility of SKU_A1 and SKU_A2
            if p['sku'] in ['SKU_A1', 'SKU_A2'] and week_idx >= n_weeks_before:
                # Cannibalization penalty (reduces their attractiveness)
                cannibalization_effect = -0.35
                utility += cannibalization_effect

            # --- KEY: Model Incrementality Effect ---
            # When SKU_A3 is present, reduce utility of competitor products (but less than own products)
            if p['sku'] in ['SKU_B1', 'SKU_B2', 'SKU_C1'] and week_idx >= n_weeks_before:
                # Incrementality effect (competitor products become less attractive)
                incrementality_effect = -0.20
                utility += incrementality_effect

            available_products.append({
                'Date': date,
                'Week': week_idx + 1,
                'SKU': p['sku'],
                'Brand': p['brand'],
                'Price': current_price,
                'ACV': acv,
                'Utility': utility,
                'Is_New_Product': 1 if p['sku'] == 'SKU_A3' else 0,
                'Post_Launch': 1 if week_idx >= n_weeks_before else 0
            })
            utilities.append(utility)

        # Calculate market shares using softmax (discrete choice model)
        probs = softmax(utilities)

        # Assign sales volumes
        for i, item in enumerate(available_products):
            # Sales = Market Size * Probability * ACV + noise
            expected_sales = market_size * probs[i] * item['ACV']

            # Add Poisson noise for realistic count data
            actual_sales = np.random.poisson(expected_sales)

            item['Sales_Volume'] = actual_sales

            # Remove utility (not observed in real data)
            del item['Utility']

            all_rows.append(item)

    df = pd.DataFrame(all_rows)

    # Create unbalanced panel: randomly drop ~10% of observations
    mask = np.random.rand(len(df)) > 0.10
    df_final = df[mask].copy()

    return df_final

# Generate the data
print("Generating new product launch simulation data...")
df = generate_new_product_launch_data(n_weeks_before=52, n_weeks_after=52)

# Save to CSV
output_file = 'new_product_launch_data.csv'
df.to_csv(output_file, index=False)

print(f"\n✓ Data saved to {output_file}")
print(f"\nDataset Shape: {df.shape}")
print(f"\nColumn Names: {list(df.columns)}")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print("\n1. Product Portfolio:")
print(df.groupby('Brand')['SKU'].unique())

print("\n2. Time Period:")
print(f"   Start Date: {df['Date'].min()}")
print(f"   End Date: {df['Date'].max()}")
print(f"   Total Weeks: {df['Week'].max()}")

print("\n3. New Product Launch (SKU_A3):")
new_product_data = df[df['SKU'] == 'SKU_A3']
if len(new_product_data) > 0:
    print(f"   Launch Week: {new_product_data['Week'].min()}")
    print(f"   Total Sales (post-launch): {new_product_data['Sales_Volume'].sum():,}")
    print(f"   Avg Weekly Sales: {new_product_data['Sales_Volume'].mean():.0f}")

print("\n4. Sales by Brand (Pre vs Post Launch):")
pre_launch = df[df['Post_Launch'] == 0].groupby('Brand')['Sales_Volume'].sum()
post_launch = df[df['Post_Launch'] == 1].groupby('Brand')['Sales_Volume'].sum()

comparison = pd.DataFrame({
    'Pre_Launch_Total': pre_launch,
    'Post_Launch_Total': post_launch,
})
comparison['Change'] = comparison['Post_Launch_Total'] - comparison['Pre_Launch_Total']
comparison['Change_Pct'] = (comparison['Change'] / comparison['Pre_Launch_Total'] * 100).round(1)
print(comparison)

print("\n5. Impact on Individual SKUs (Avg Weekly Sales):")
pre_sku = df[df['Post_Launch'] == 0].groupby('SKU')['Sales_Volume'].mean()
post_sku = df[df['Post_Launch'] == 1].groupby(['SKU', 'Is_New_Product'])['Sales_Volume'].mean().reset_index()

# Separate new product
existing_post = post_sku[post_sku['Is_New_Product'] == 0].set_index('SKU')['Sales_Volume']

impact = pd.DataFrame({
    'Pre_Launch_Avg': pre_sku,
    'Post_Launch_Avg': existing_post,
})
impact['Change'] = impact['Post_Launch_Avg'] - impact['Pre_Launch_Avg']
impact['Change_Pct'] = (impact['Change'] / impact['Pre_Launch_Avg'] * 100).round(1)
print(impact)

print("\n6. Sample of data (first 10 rows):")
print(df.head(10).to_string(index=False))

print("\n" + "="*70)
print("Next Steps:")
print("="*70)
print("Use PyMC to estimate:")
print("1. Price elasticity")
print("2. ACV effect")
print("3. Cannibalization effect (impact on Brand_A SKUs)")
print("4. Incrementality effect (impact on Brand_B and Brand_C)")
print("5. True incremental sales from SKU_A3")
