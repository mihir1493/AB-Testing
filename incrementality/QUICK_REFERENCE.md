# Quick Reference: Key Concepts & Interpretations

## The Core Question
**SKU_A3 sold 951K units. Where did those sales come from?**

---

## Data Structure

```
Week | SKU    | Brand   | Price | ACV  | Sales | Post_Launch | Cannibalized | Incremental
-----|--------|---------|-------|------|-------|-------------|--------------|------------
50   | SKU_A1 | Brand_A | 2.00  | 0.95 | 28000 | 0           | 0            | 0
55   | SKU_A1 | Brand_A | 1.98  | 0.92 | 19500 | 1           | 1            | 0  ← Cannibalized
55   | SKU_B1 | Brand_B | 1.90  | 0.88 | 20000 | 1           | 0            | 1  ← Incremental source
55   | SKU_A3 | Brand_A | 2.30  | 0.50 | 15000 | 1           | 0            | 0  ← New product
```

---

## Model Equation

```
log(Sales) = α[SKU] + β_price × Price + β_acv × ACV + β_cannib × Cannibalized + β_increm × Incremental + ε
             ────┬───   ─────┬──────   ────┬─────   ──────────┬──────────   ────────┬────────
                 │           │             │                   │                     │
         Product baseline   Price      Availability    Own products hurt    Competitors hurt
                            effect        effect       when new product     when new product
                                                            launches            launches
```

---

## Key Parameters & Interpretation

| Parameter | What it measures | Expected sign | How to interpret |
|-----------|-----------------|---------------|------------------|
| `α[SKU]` | Product baseline appeal | Any | α[A1]=9.8 means SKU_A1 has higher baseline than α[B1]=9.5 |
| `β_price` | Price elasticity | **Negative** | β=-0.42 → 1-SD ↑ price = exp(-0.42)-1 = -34% sales |
| `β_acv` | Distribution effect | **Positive** | β=+0.51 → 1-SD ↑ ACV = exp(0.51)-1 = +67% sales |
| `β_cannib` | **Cannibalization** | **Negative** | β=-0.34 → Own products decline 29% when new product launches |
| `β_increm` | **Incrementality** | **Negative** | β=-0.19 → Competitors decline 17% when new product launches |

---

## Converting Coefficients to Business Metrics

### From model coefficient to percentage change:
```python
beta_cannib = -0.341
percentage_change = (np.exp(beta_cannib) - 1) * 100
# = -28.9%

→ "Own products declined 29% after controlling for price/ACV"
```

### From percentage change to volume impact:
```python
# SKU_A1 pre-launch average: 28,000 units/week
# Impact: -29%
volume_loss = 28000 * 0.29 = 8,120 units/week

# Over 52 weeks post-launch:
total_cannibalized = 8120 * 52 = 422,240 units
```

---

## Credible Intervals (Uncertainty)

```
beta_cannib: mean=-0.341, 95% CI=[-0.461, -0.221]
                                   ─────┬──────
                                         │
                                    We're 95% confident
                                    true effect is in
                                    this range
```

**What this means:**
- Best estimate: -0.341 (29% decline)
- Worst case: -0.461 (37% decline)
- Best case: -0.221 (20% decline)
- **It's definitely negative** (entire interval < 0)

---

## Volume Decomposition Calculation

### Step 1: Calculate total impact on own products
```python
# Pre-launch: Sum of all Brand_A sales in weeks 1-52
pre_own = 1,679,749 units

# Post-launch: Sum of Brand_A sales (excluding SKU_A3) in weeks 53-104
post_own = 1,194,695 units

# Cannibalized volume
cannibalized = pre_own - post_own = 485,054 units
```

### Step 2: Calculate total impact on competitors
```python
pre_comp = 2,502,580 units
post_comp = 1,954,867 units

incremental = pre_comp - post_comp = 547,713 units
```

### Step 3: Calculate rates
```python
total_new_sales = 951,471 units

cannibalization_rate = 485,054 / 951,471 = 51.0%
incrementality_rate = 547,713 / 951,471 = 57.6%

# Why > 100%? Market dynamics + measurement error
```

---

## Model Diagnostics Checklist

| Metric | Good Value | What it means | Action if bad |
|--------|-----------|---------------|---------------|
| **R-hat** | < 1.01 | Chains converged | Increase tune/draws |
| **ESS** | > 1000 | Enough independent samples | Increase draws |
| **R²** | > 0.7 | Model fits data well | Add variables or interactions |
| **Residuals** | Random scatter | No systematic bias | Check for non-linearity |

---

## Common Business Questions & How to Answer Them

### Q1: "How much did SKU_A3 cannibalize our existing products?"
**Answer using `beta_cannib`:**
```
Model shows β_cannib = -0.341 [95% CI: -0.461, -0.221]
→ Own products declined 29% [95% CI: 20-37%]
→ Translates to ~485K units cannibalized out of 951K total
→ Cannibalization rate: 51%
```

### Q2: "Did we steal share from competitors?"
**Answer using `beta_increm`:**
```
Model shows β_increm = -0.189 [95% CI: -0.296, -0.083]
→ Competitors declined 17% [95% CI: 8-26%]
→ Translates to ~548K units from competitors
→ Incrementality rate: 58%
```

### Q3: "What was the net impact on our brand?"
**Answer using volume decomposition:**
```
Total SKU_A3 sales: 951K
Cannibalized: 485K
Net incremental to brand: 951K - 485K = 466K units

→ "For every 2 units of SKU_A3 sold, 1 was truly incremental to the brand"
```

### Q4: "Should we have launched this product?"
**Answer using profitability:**
```
Assumptions:
- SKU_A3 margin: $0.50/unit
- Cannibalized products average margin: $0.45/unit

SKU_A3 gross profit: 951K × $0.50 = $475,550
Lost profit from cannibalization: 485K × $0.45 = -$218,250
Net profit impact: $475,550 - $218,250 = $257,300

→ "Yes, net positive $257K in gross profit"
```

---

## Warning Signs in Results

### 🚨 High Cannibalization (> 60%)
**What it means:** New product is mostly stealing from your own products

**Actions:**
- Reassess product positioning
- Consider discontinuing lowest-margin cannibalized SKU
- Differentiate new product more clearly
- Adjust relative pricing

### 🚨 Low Incrementality (< 30%)
**What it means:** Not stealing much from competitors

**Actions:**
- Analyze competitive dynamics—are they not substitutes?
- Check if competitor decline is statistically significant
- Consider if market is growing (incrementality could be new customers)

### 🚨 Cannibalization + Incrementality >> 100%
**What it means:** Your model is over-attributing decline to the new product

**Possible causes:**
- Market contraction not captured
- Seasonal effects not properly controlled
- Missing confounders (e.g., competitor promotions)

**Fix:**
- Add time trend or seasonality terms
- Include competitor pricing
- Use longer pre-period data

---

## Visualizations Cheat Sheet

### 1. Time Series Plot
**Purpose:** Show the launch moment and visual impact
```python
ax.plot(weeks, sales_sku_a1, label='SKU_A1 (Own)')
ax.plot(weeks, sales_sku_b1, label='SKU_B1 (Competitor)')
ax.axvline(x=53, color='red', label='Launch')
```
**What to look for:** Clear change in trend at launch week

---

### 2. Pre/Post Bar Chart
**Purpose:** Quantify average sales change
```python
ax.bar(x - width/2, pre_avg, label='Pre-Launch')
ax.bar(x + width/2, post_avg, label='Post-Launch')
```
**What to look for:** All existing products should decline

---

### 3. Posterior Distribution Histograms
**Purpose:** Show uncertainty in key parameters
```python
ax.hist(trace.posterior['beta_cannib'].values.flatten(), bins=50)
ax.axvline(mean, color='red', label='Mean')
ax.axvline(ci_lower, color='orange', label='95% CI')
```
**What to look for:**
- Distribution shouldn't cross zero (confirms effect is real)
- Narrow distribution = high confidence

---

### 4. Pie Chart (Sales Decomposition)
**Purpose:** Show where new product sales came from
```python
sizes = [cannibalized_volume, incremental_volume]
labels = ['Cannibalized', 'Incremental']
ax.pie(sizes, labels=labels)
```
**What to look for:** Bigger green slice (incremental) is better

---

## Bayesian vs Frequentist: Quick Comparison

| Aspect | Frequentist | Bayesian (Our Approach) |
|--------|------------|------------------------|
| **Output** | Point estimate + p-value | Full probability distribution |
| **Uncertainty** | Confidence interval | Credible interval |
| **Interpretation** | "If we repeated this 100 times, 95 would contain true value" | "95% probability true value is in this range" |
| **Priors** | None | Incorporates domain knowledge |
| **Question answered** | "Is effect different from zero?" | "What is the most likely effect size?" |

**Why Bayesian for this problem:**
- We have prior beliefs about price elasticity (should be negative)
- We want probabilistic statements ("70% chance cannibalization > 25%")
- Handles complex hierarchical structure (products nested in brands)

---

## Sanity Checks

✅ **Price coefficient should be negative**
- If positive → model is broken or severe multicollinearity

✅ **ACV coefficient should be positive**
- If negative → check ACV is coded correctly (not inverted)

✅ **Cannibalization should be stronger than incrementality** (usually)
- Own products are closer substitutes than competitor products
- If β_increm < β_cannib → surprising, investigate

✅ **R-hat ≈ 1.0 for all parameters**
- If any R-hat > 1.01 → don't trust results, re-run with more samples

✅ **Residuals should be random**
- If pattern in residuals → missing variable or wrong functional form

---

## Next Steps After Analysis

### For Management Presentation:
1. **Executive Summary Slide:**
   - New product sales: 951K units
   - Net incremental: 466K units (49%)
   - ROI calculation with profit margins

2. **Detail Slides:**
   - Time series showing launch impact
   - Pie chart of sales decomposition
   - Strategic recommendations

### For Strategic Planning:
1. **Portfolio Optimization:**
   - Which SKUs to keep/discontinue
   - Optimal product mix

2. **Future Launch Framework:**
   - Use these cannibalization rates as benchmarks
   - Set thresholds for launch decisions

3. **Competitive Intelligence:**
   - Which competitors we're stealing from most
   - Where to focus marketing

---

## Code Snippets for Common Tasks

### Get probability of threshold
```python
samples = trace.posterior['beta_cannib'].values.flatten()
prob_worse_than_30pct = (samples < np.log(0.7)).mean()
print(f"Probability cannibalization > 30%: {prob_worse_than_30pct:.1%}")
```

### Calculate expected volume impact
```python
# Expected cannibalization per week
beta_cannib_mean = samples.mean()
baseline_sales = 28000  # SKU_A1 pre-launch average
expected_loss = baseline_sales * (np.exp(beta_cannib_mean) - 1)
print(f"Expected weekly loss: {expected_loss:.0f} units")
```

### Convert coefficients to elasticities
```python
# Price elasticity (percent change in sales per 1% change in price)
beta_price = -0.423
price_mean = df['Price'].mean()
# Need to multiply by price/sales ratio for true elasticity
```

---

## Troubleshooting

### "MCMC didn't converge (R-hat > 1.01)"
- Increase `tune` to 2000
- Increase `target_accept` to 0.99
- Check for data issues (extreme outliers)
- Rescale variables if very different magnitudes

### "ESS is very low (< 400)"
- Increase `draws` to 4000
- Check for multicollinearity (highly correlated predictors)
- Try different parameterization

### "Model predicts negative sales"
- Use Poisson or NegativeBinomial likelihood instead of Normal
- Or: Ensure log-transform is applied correctly

### "Coefficients don't make sense (price is positive)"
- Check for multicollinearity: `df[['price_std', 'acv_std']].corr()`
- Check data entry errors
- Consider interaction terms

---

## Resources

- **Full notebook:** `incrementality_analysis.ipynb`
- **Detailed guide:** `STEP_BY_STEP_GUIDE.md`
- **Data generation:** `new_product_launch_simulation.py`
- **Data file:** `new_product_launch_data.csv`
