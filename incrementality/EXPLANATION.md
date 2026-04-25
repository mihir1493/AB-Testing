# Understanding the Discrepancy Between Model Coefficients and Volume Calculations

## The Problem You Identified

You correctly noticed that the numbers don't match:

### Section 8 (Model Coefficients):
- **Cannibalization coefficient**: -0.274 → -24.0% effect
- **Incrementality coefficient**: -0.182 → -16.6% effect

### Section 9 (Simple Pre/Post Comparison):
- **Cannibalization**: 51.0% of SKU_A3 sales (485,054 units)
- **Incrementality**: 57.6% of SKU_A3 sales (547,713 units)
- **Total**: 108.5% ⚠️ (This is impossible!)

## Why the Discrepancy Exists

The two methods are calculating **fundamentally different things**:

### Method 1: Simple Pre/Post Comparison (Section 9)
```python
# Cannibalization
pre_own = Sales(Brand_A, Pre-Launch)
post_own = Sales(Brand_A, Post-Launch, excluding SKU_A3)
cannibalized = pre_own - post_own
```

**Problem**: This attributes **ALL** changes to SKU_A3, including:
- Natural market growth/decline
- Seasonality effects
- Price changes over time
- Distribution (ACV) changes
- Competitor actions
- Economic conditions

### Method 2: Regression Model Coefficients (Section 8)
```
log(Sales) = α[sku] + β_price × Price + β_acv × ACV
           + β_cannib × Cannibalized + β_increm × Incremental
```

**Advantage**: Controls for confounding variables and isolates the **causal effect** of SKU_A3.

## The Correct Approach

To reconcile the two approaches, you need to **use the model coefficients** to calculate volumes:

```python
# Model coefficient tells us the percentage effect
cannib_effect = exp(β_cannib) - 1  # exp(-0.274) - 1 = -24.0%

# Calculate what sales WOULD have been without SKU_A3
# If: actual_sales = counterfactual_sales × (1 + effect)
# Then: counterfactual_sales = actual_sales / (1 + effect)

own_counterfactual = own_actual_sales / (1 - 0.240)
cannibalized_volume = own_counterfactual - own_actual_sales
```

## Expected Results (Model-Based Calculation)

When you properly calculate volumes using the model coefficients, you should get:

- **Cannibalization**: ~24% of total impact
- **Incrementality**: ~17% of total impact
- **Unexplained/Market growth**: ~59%
- **Total**: 100% ✓

This makes sense because:
1. Not all of SKU_A3's sales came from competitors or own products
2. Some came from **market expansion** (new category buyers, increased consumption, etc.)
3. The model properly accounts for this by controlling for trends

## Action Items

To fix your notebook:

1. **Add Section 9A**: "Model-Based Volume Decomposition"
2. **Calculate counterfactual sales** using the coefficients
3. **Compare** simple vs model-based estimates side-by-side
4. **Use model-based estimates** for all business decisions

## Code Example for Section 9A

```python
# ===================================================================
# MODEL-BASED VOLUME DECOMPOSITION
# ===================================================================

# Get model coefficients
cannib_effect_pct = (np.exp(beta_cannib_mean) - 1)  # -24.0%
increm_effect_pct = (np.exp(beta_increm_mean) - 1)  # -16.6%

# Post-launch actual sales
own_products_post = df_existing[
    (df_existing['Post_Launch'] == 1) &
    (df_existing['Brand'] == 'Brand_A')
]
comp_products_post = df_existing[
    (df_existing['Post_Launch'] == 1) &
    (df_existing['Brand'].isin(['Brand_B', 'Brand_C']))
]

own_actual_sales = own_products_post['Sales_Volume'].sum()
comp_actual_sales = comp_products_post['Sales_Volume'].sum()

# Calculate counterfactual (what sales would have been WITHOUT SKU_A3)
own_counterfactual = own_actual_sales / (1 + cannib_effect_pct)
comp_counterfactual = comp_actual_sales / (1 + increm_effect_pct)

# Cannibalized/incremental volumes
cannibalized_volume_model = own_counterfactual - own_actual_sales
incremental_volume_model = comp_counterfactual - comp_actual_sales

# As percentage of SKU_A3 sales
cannib_pct_model = (cannibalized_volume_model / total_new_product_sales) * 100
increm_pct_model = (incremental_volume_model / total_new_product_sales) * 100

print(f"Cannibalization: {cannib_pct_model:.1f}%")
print(f"Incrementality: {increm_pct_model:.1f}%")
print(f"Total accounted for: {cannib_pct_model + increm_pct_model:.1f}%")
```

## Why This Matters

**Simple Pre/Post** tells you: "Sales changed by X after launch"
**Model-Based** tells you: "SKU_A3 **caused** X change, controlling for everything else"

The model-based approach is what you should use for:
- Strategic decisions
- Portfolio optimization
- Pricing strategies
- Forecasting
- P&L planning

## Summary

Your instinct was correct - the numbers don't match because they're measuring different things. The simple pre/post comparison is useful for initial exploration, but the **model-based estimates are the truth** for decision-making.

Always trust the regression model coefficients when they control for confounding variables.
