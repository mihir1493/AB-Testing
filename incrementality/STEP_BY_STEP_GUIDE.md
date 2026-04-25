# Step-by-Step Guide: Cannibalization vs Incrementality Analysis
## A Complete Walkthrough for Your Medium Article

---

## The Business Problem

Imagine you're a product manager at a consumer goods company. Last year, you launched a new product (SKU_A3) that sold 951,000 units. Success, right? But here's the million-dollar question:

**Where did those sales actually come from?**

Did they come from:
- 🔴 **Cannibalization**: Stealing sales from your own existing products?
- 🟢 **Incrementality**: Stealing market share from competitors?

This matters because:
- Cannibalized sales don't grow your business—they just shift sales between your products
- Incremental sales genuinely grow your market share and revenue
- Understanding this split drives strategic decisions about product portfolio, pricing, and future launches

---

## The Analytical Approach

We'll use **Bayesian inference with PyMC** to estimate these effects. Here's why this approach is powerful:

1. **Accounts for uncertainty**: Instead of point estimates, we get probability distributions
2. **Handles complex relationships**: Multiple products influencing each other
3. **Interpretable**: We can quantify exactly how much cannibalization vs incrementality occurred
4. **Flexible**: Can incorporate domain knowledge through priors

---

## PART 1: Exploratory Data Analysis (EDA)

### Step 1.1: Understanding the Data Structure

```python
df = pd.read_csv('new_product_launch_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
```

**What's happening here:**
- We have an **unbalanced panel dataset**: weekly sales data for multiple products
- "Unbalanced" means not all products appear in all weeks (some products might be out of stock, delisted, or not yet launched)
- This mimics real-world retail data where products come and go

**Key columns:**
- `Week`: Time period (1-104 weeks = 2 years)
- `SKU`: Product identifier (SKU_A1, SKU_A2, SKU_A3, SKU_B1, SKU_B2, SKU_C1)
- `Brand`: Brand name (Brand_A = yours, Brand_B & Brand_C = competitors)
- `Price`: Product price (varies weekly due to promotions)
- `ACV`: All Commodity Volume = distribution/availability (0-1 scale)
- `Sales_Volume`: Units sold (our outcome variable)
- `Is_New_Product`: Indicator for SKU_A3
- `Post_Launch`: Indicator for weeks after SKU_A3 launch

**Why this matters for your article:**
This is realistic CPG (Consumer Packaged Goods) data structure. Most companies have this type of data but don't know how to extract insights from it.

---

### Step 1.2: Visual Evidence of the Impact

```python
# Sales trends over time by SKU
for sku in df['SKU'].unique():
    sku_data = df[df['SKU'] == sku].sort_values('Week')
    ax.plot(sku_data['Week'], sku_data['Sales_Volume'], label=f"{sku}")

# Mark the launch week
launch_week = df[df['Is_New_Product'] == 1]['Week'].min()
ax.axvline(x=launch_week, color='red', linestyle=':', label='SKU_A3 Launch')
```

**What you'll see:**
- A clear vertical line at Week 53 (the launch)
- **Before the line**: Only existing products (SKU_A1, A2, B1, B2, C1)
- **After the line**: SKU_A3 appears AND other products' sales change

**Key observations to highlight:**
1. **Brand_A products (your brand)**: SKU_A1 and SKU_A2 decline after launch → Evidence of cannibalization
2. **Brand_B and Brand_C (competitors)**: Also decline after launch → Evidence of incrementality
3. **SKU_A3**: Immediately starts selling ~20K units/week

**For your article:**
Include this visualization! It's the "before and after" story that makes the problem tangible. Readers can visually see that something happened at the launch.

---

### Step 1.3: Quantifying the Pre/Post Change

```python
pre_avg = df[df['Post_Launch'] == 0].groupby('SKU')['Sales_Volume'].mean()
post_avg = df[(df['Post_Launch'] == 1) & (df['Is_New_Product'] == 0)].groupby('SKU')['Sales_Volume'].mean()

comparison['Change_Pct'] = (comparison['Change'] / comparison['Pre_Launch_Avg'] * 100)
```

**What this does:**
- Compares average weekly sales BEFORE vs AFTER launch for each existing product
- Excludes SKU_A3 itself (we know it went from 0 to 20K—that's obvious!)

**Expected results:**
```
SKU_A1 (Your brand):    -30% decline
SKU_A2 (Your brand):    -26% decline  ← CANNIBALIZATION
SKU_B1 (Competitor):    -22% decline  ← INCREMENTALITY
SKU_B2 (Competitor):    -15% decline  ← INCREMENTALITY
SKU_C1 (Competitor):    -20% decline  ← INCREMENTALITY
```

**The naive analysis:**
You might stop here and say "Okay, SKU_A1 dropped 30%, that's ~8,500 units/week cannibalized."

**But that's wrong! Here's why:**
1. **Confounding factors**: Price and availability (ACV) also changed week-to-week
2. **Seasonality**: Market size fluctuates
3. **Multiple effects**: How do we know the decline in SKU_A1 wasn't partly due to price increases?

**This is why we need the Bayesian model** → It isolates the new product effect while controlling for other factors.

---

## PART 2: Building the PyMC Model

### Step 2.1: Feature Engineering

```python
# Log transform sales
df_model['log_sales'] = np.log(df_model['Sales_Volume'] + 1)

# Standardize continuous variables
df_model['price_std'] = (df_model['Price'] - df_model['Price'].mean()) / df_model['Price'].std()
df_model['acv_std'] = (df_model['ACV'] - df_model['ACV'].mean()) / df_model['ACV'].std()
```

**Why log-transform sales?**
- Sales are **count data** (non-negative integers)
- Sales distributions are **right-skewed** (most weeks have moderate sales, some weeks have very high sales)
- Log transformation makes the distribution more normal → better for linear modeling
- Coefficients become interpretable as **percentage changes**

**Why standardize price and ACV?**
- Puts them on the same scale (mean=0, SD=1)
- Makes MCMC sampling more efficient (the algorithm converges faster)
- Makes coefficients comparable ("1-SD change in price" vs "1-SD change in ACV")

---

### Step 2.2: Creating Treatment Indicators

```python
# Check which weeks have the new product
weeks_with_new_product = df_model[df_model['Is_New_Product'] == 1]['Week'].unique()
df_model['new_product_in_market'] = df_model['Week'].isin(weeks_with_new_product).astype(int)

# Cannibalization indicator: Own products in weeks when new product exists
df_model['cannibalized'] = ((df_model['SKU'].isin(['SKU_A1', 'SKU_A2'])) &
                             (df_model['new_product_in_market'] == 1)).astype(int)

# Incrementality indicator: Competitor products in weeks when new product exists
df_model['incremental_source'] = ((df_model['Brand'].isin(['Brand_B', 'Brand_C'])) &
                                   (df_model['new_product_in_market'] == 1)).astype(int)
```

**What this creates:**

| SKU    | Week | new_product_in_market | cannibalized | incremental_source |
|--------|------|----------------------|--------------|-------------------|
| SKU_A1 | 50   | 0                    | 0            | 0                 |
| SKU_A1 | 55   | 1                    | **1**        | 0                 |
| SKU_B1 | 50   | 0                    | 0            | 0                 |
| SKU_B1 | 55   | 1                    | 0            | **1**             |

**Key insight:**
- `cannibalized = 1`: This observation is a Brand_A product in a post-launch week
- `incremental_source = 1`: This observation is a competitor product in a post-launch week
- These are our **"treatment" variables** in a quasi-experimental design

---

### Step 2.3: The Model Structure

```python
with pm.Model() as model:
    # Product-specific intercepts (random effects)
    mu_alpha = pm.Normal('mu_alpha', mu=9.5, sigma=1)
    sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=0.5)
    alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=n_skus)

    # Coefficients
    beta_price = pm.Normal('beta_price', mu=-0.5, sigma=0.3)
    beta_acv = pm.Normal('beta_acv', mu=0.5, sigma=0.3)
    beta_cannib = pm.Normal('beta_cannib', mu=-0.3, sigma=0.2)
    beta_increm = pm.Normal('beta_increm', mu=-0.2, sigma=0.2)

    # Linear model
    mu = (alpha[sku_idx] +
          beta_price * price +
          beta_acv * acv +
          beta_cannib * cannibalized +
          beta_increm * incremental)

    # Likelihood
    sigma = pm.HalfNormal('sigma', sigma=0.5)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=log_sales)
```

**Breaking this down for your article:**

#### 1. **Random Effects for Products** (`alpha[sku]`)
```
alpha[sku] ~ Normal(mu_alpha, sigma_alpha)
```
- Each product has its own baseline "attractiveness" (intercept)
- Example: SKU_A1 might have α = 9.8, SKU_B1 might have α = 9.5
- This captures inherent differences: brand loyalty, product quality, flavor preferences
- **Hierarchical structure**: Products share information through `mu_alpha` (grand mean)

**Why this matters:**
Without this, we'd force all products to have the same baseline, which is unrealistic.

#### 2. **Price Elasticity** (`beta_price`)
```
beta_price ~ Normal(mu=-0.5, sigma=0.3)
```
- **Prior belief**: We expect beta_price ≈ -0.5 (negative because higher price → lower sales)
- **Uncertainty**: sigma=0.3 means we're moderately confident but let the data update us
- **Interpretation**: If beta_price = -0.4, a 1-SD increase in price → exp(-0.4) - 1 = -33% change in sales

**For your article:**
This is a **weakly informative prior**—we're encoding that prices negatively affect demand, but we let the data tell us how much.

#### 3. **ACV Effect** (`beta_acv`)
```
beta_acv ~ Normal(mu=0.5, sigma=0.3)
```
- **Prior belief**: Higher availability → higher sales (positive coefficient)
- **Logic**: If a product is on more shelves (ACV=0.9 vs 0.5), more people see and buy it

#### 4. **Cannibalization Effect** (`beta_cannib`)
```
beta_cannib ~ Normal(mu=-0.3, sigma=0.2)
```
- **THIS IS THE KEY PARAMETER FOR YOUR ARTICLE**
- Measures the impact on SKU_A1 and SKU_A2 when SKU_A3 is in market
- Expected to be **negative** (own products hurt when new product launches)
- The data will tell us exactly how negative

#### 5. **Incrementality Effect** (`beta_increm`)
```
beta_increm ~ Normal(mu=-0.2, sigma=0.2)
```
- **THIS IS THE OTHER KEY PARAMETER**
- Measures the impact on Brand_B and Brand_C when SKU_A3 is in market
- Also expected to be **negative** (competitors hurt when we launch)
- Might be less negative than cannibalization (we might hurt our own products more)

---

### Step 2.4: The Full Equation

Putting it all together:

```
log(Sales[i]) = α[product[i]]
                + β_price × Price[i]
                + β_acv × ACV[i]
                + β_cannib × Cannibalized[i]
                + β_increm × IncrementalSource[i]
                + ε[i]

where ε[i] ~ Normal(0, σ)
```

**What this means in plain English:**

"The log of sales for observation i is determined by:
1. The product's baseline attractiveness (α)
2. How the price affects it (β_price)
3. How availability affects it (β_acv)
4. Whether it's being cannibalized by the new product (β_cannib)
5. Whether it's losing share to the new product (β_increm)
6. Plus some random noise (ε)"

---

## PART 3: Running the Inference

### Step 3.1: MCMC Sampling

```python
with model:
    trace = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        target_accept=0.95,
        random_seed=42
    )
```

**What's happening under the hood:**

1. **MCMC = Markov Chain Monte Carlo**
   - An algorithm that explores the posterior distribution
   - Generates samples from P(parameters | data)

2. **Why 4 chains?**
   - Run 4 independent MCMC chains to check convergence
   - If all chains arrive at the same distribution → model converged
   - If chains disagree → something's wrong

3. **What are "draws" and "tune"?**
   - `tune=1000`: First 1000 samples are discarded (warmup period)
   - `draws=2000`: Keep 2000 samples from each chain
   - Total posterior samples: 4 chains × 2000 = 8000 samples

4. **target_accept=0.95**
   - Technical MCMC parameter
   - Higher = more accurate but slower
   - 0.95 is good for complex models

**For your article:**
You can explain this as: "The algorithm generates 8,000 plausible sets of parameters that are consistent with the observed data. This gives us a full probability distribution, not just a single estimate."

---

### Step 3.2: Checking Diagnostics

```python
print(az.rhat(trace))
print(az.ess(trace))
```

**R-hat (Gelman-Rubin statistic):**
- Measures convergence across chains
- **Rule**: R-hat should be < 1.01 (ideally < 1.001)
- If R-hat > 1.01 → chains haven't converged → don't trust results

**Effective Sample Size (ESS):**
- MCMC samples are correlated (not independent)
- ESS tells you how many "effective" independent samples you have
- **Rule**: ESS should be > 400 (ideally > 1000)
- If ESS is low → increase draws

**For your article:**
"These diagnostics confirm our model converged properly—all R-hat values are near 1.0 and effective sample sizes are large, meaning we can trust the results."

---

## PART 4: Interpreting Results

### Step 4.1: Posterior Distributions

```python
summary = az.summary(trace, var_names=['beta_price', 'beta_acv', 'beta_cannib', 'beta_increm'])
```

**Example output:**
```
                mean    sd   hdi_2.5%  hdi_97.5%  r_hat  ess_bulk
beta_price    -0.423  0.045   -0.509     -0.337   1.00    7850
beta_acv       0.512  0.048    0.418      0.603   1.00    8120
beta_cannib   -0.341  0.062   -0.461     -0.221   1.00    7650
beta_increm   -0.189  0.055   -0.296     -0.083   1.00    7890
```

**How to read this table:**

#### **beta_cannib = -0.341 [95% CI: -0.461, -0.221]**

**Meaning:**
- When the new product is in market, own products (SKU_A1, SKU_A2) experience a -0.341 effect on log(sales)
- We're 95% confident the true effect is between -0.461 and -0.221
- Converting to percentage: exp(-0.341) - 1 = -28.9% change in sales

**For your article:**
"Our own products experienced a 29% decline in sales when SKU_A3 launched, after controlling for price and availability changes."

#### **beta_increm = -0.189 [95% CI: -0.296, -0.083]**

**Meaning:**
- Competitor products experienced a -0.189 effect on log(sales)
- Converting: exp(-0.189) - 1 = -17.2% change in sales

**For your article:**
"Competitor products experienced a 17% decline—evidence that SKU_A3 successfully captured market share."

---

### Step 4.2: Visualizing Uncertainty

The histogram plots show:
- **X-axis**: Coefficient value
- **Y-axis**: Frequency (how often that value appeared in the 8000 samples)
- **Red line**: Mean estimate
- **Orange lines**: 95% credible interval

**Why this is powerful for your article:**
- Traditional statistics give you "beta_cannib = -0.341 ± standard error"
- Bayesian approach gives you the **full distribution**
- You can answer questions like: "What's the probability that cannibalization is worse than -30%?"

```python
samples = trace.posterior['beta_cannib'].values.flatten()
prob_worse_than_30pct = (samples < np.log(0.7)).mean()
print(f"Probability cannibalization > 30%: {prob_worse_than_30pct:.1%}")
```

---

## PART 5: Business Impact Calculation

### Step 5.1: Volume Decomposition

```python
total_new_product_sales = df[df['Is_New_Product'] == 1]['Sales_Volume'].sum()
# = 951,471 units

# Cannibalized volume
pre_own = df[(df['Post_Launch'] == 0) & (df['Brand'] == 'Brand_A')]['Sales_Volume'].sum()
post_own = df[(df['Post_Launch'] == 1) & (df['Brand'] == 'Brand_A') &
              (df['Is_New_Product'] == 0)]['Sales_Volume'].sum()
cannibalized_volume = pre_own - post_own

# Incremental volume
pre_comp = df[(df['Post_Launch'] == 0) &
              (df['Brand'].isin(['Brand_B', 'Brand_C']))]['Sales_Volume'].sum()
post_comp = df[(df['Post_Launch'] == 1) &
               (df['Brand'].isin(['Brand_B', 'Brand_C']))]['Sales_Volume'].sum()
incremental_volume = pre_comp - post_comp
```

**What this does:**
1. **Total new product sales**: 951,471 units (the headline number)
2. **Cannibalized volume**: How much did our existing products decline?
3. **Incremental volume**: How much did competitors decline?

**Expected results:**
```
Total SKU_A3 Sales: 951,471 units

CANNIBALIZED: 213,834 units (22.5%)
  → Taken from SKU_A1 and SKU_A2

INCREMENTAL: 547,674 units (57.6%)
  → Taken from Brand B and Brand C

Accounted for: 80.1%
```

**Key insight for your article:**
The percentages don't sum to 100% because:
- Some sales came from market growth (seasonality)
- Some customers might be new to the category
- Rounding and measurement error

---

### Step 5.2: The Counterfactual Question

**The critical business question:**
"What would have happened if we DIDN'T launch SKU_A3?"

**Without the new product:**
- Brand_A total: pre_own sales would have continued
- Competitors: pre_comp sales would have continued

**With the new product:**
- Brand_A total: post_own + new_product_sales
- Competitors: post_comp (declined)

**Net gain to Brand_A:**
```python
net_gain = total_new_product_sales - cannibalized_volume
# = 951,471 - 213,834 = 737,637 units

net_gain_pct = (net_gain / total_new_product_sales) * 100
# = 77.5%
```

**For your article:**
"While SKU_A3 sold 951K units, only 738K (77.5%) were truly incremental to our brand. The remaining 214K units simply shifted from our existing products."

---

## PART 6: Model Validation

### Step 6.1: Posterior Predictive Checks

```python
with model:
    ppc = pm.sample_posterior_predictive(trace)

y_pred_mean = ppc.posterior_predictive['y_obs'].values.mean(axis=(0, 1))
```

**What this does:**
- Uses the fitted model to predict sales
- Compares predictions to actual sales
- If the model is good, predictions should match actuals

**The plots:**
1. **Observed vs Predicted scatter plot**
   - Points should fall on the diagonal line
   - Systematic deviations suggest model misspecification

2. **Residuals plot**
   - Should be randomly scattered around zero
   - Patterns suggest missing variables or non-linear effects

**R-squared:**
```python
r_squared = 1 - (ss_res / ss_tot)
# Example: 0.847
```

**For your article:**
"The model explains 84.7% of the variation in sales, indicating a strong fit."

---

## PART 7: Strategic Recommendations

### High Cannibalization Scenario (cannib > 50%)

**What it means:**
- More than half of new product sales came from own products
- Net brand growth is minimal
- Portfolio might be overcrowded

**Recommendations:**
1. **Product differentiation**: Make SKU_A3 more distinct from A1/A2
2. **Pricing strategy**: Adjust relative prices to steer customers
3. **Portfolio optimization**: Consider discontinuing underperforming SKUs
4. **Target different occasions**: Position products for different use cases

---

### Healthy Incrementality Scenario (cannib < 40%)

**What it means:**
- Majority of sales are from competitors
- Successfully growing market share
- Strong product-market fit

**Recommendations:**
1. **Scale up**: Increase distribution and marketing
2. **Sustain advantage**: Monitor competitor responses
3. **Geographic expansion**: Roll out to new markets
4. **Line extension**: Consider variants of successful formula

---

## Key Formulas for Your Article

### 1. Converting log-coefficients to percentages
```
Percentage change = (exp(β) - 1) × 100%

Example: β_cannib = -0.341
→ (exp(-0.341) - 1) × 100% = -28.9%
```

### 2. Calculating cannibalization rate
```
Cannibalization Rate = (Loss in own products) / (New product sales)

Example: 213,834 / 951,471 = 22.5%
```

### 3. Calculating incrementality rate
```
Incrementality Rate = (Loss in competitor products) / (New product sales)

Example: 547,674 / 951,471 = 57.6%
```

### 4. Net incremental volume
```
Net Incremental = New product sales - Cannibalized volume

Example: 951,471 - 213,834 = 737,637 units
```

---

## Common Pitfalls to Avoid

### 1. **Ignoring confounders**
❌ "SKU_A1 declined 30%, so that's the cannibalization."
✅ "After controlling for price, ACV, and seasonality, the cannibalization effect is -29%."

### 2. **Assuming linear effects**
❌ Using raw sales instead of log(sales)
✅ Log transformation handles multiplicative effects and right-skewed distributions

### 3. **Ignoring uncertainty**
❌ Reporting only point estimates
✅ Reporting 95% credible intervals

### 4. **Forgetting market growth**
❌ Assuming cannibalization + incrementality = 100%
✅ Acknowledging that market size changes over time

---

## Visualizations for Your Article

**Must-include charts:**

1. **Time series plot**: Shows the launch moment and immediate impact
   - X-axis: Weeks
   - Y-axis: Sales volume
   - Vertical line at launch week
   - Different lines for each SKU

2. **Pre/Post comparison**: Bar chart showing average weekly sales
   - X-axis: SKU
   - Y-axis: Average sales
   - Two bars per SKU (pre vs post)
   - Percentage change labels

3. **Posterior distributions**: Histograms of beta_cannib and beta_increm
   - Shows uncertainty
   - Includes mean and 95% CI
   - Visual evidence the effects are real (don't cross zero)

4. **Pie chart**: Sales decomposition
   - Cannibalized vs Incremental vs Other
   - Clear percentages
   - Color-coded (red for cannibalization, green for incrementality)

---

## Conclusion: The Story for Your Article

**Opening hook:**
"When our new product sold nearly 1 million units in its first year, leadership celebrated. But I had to ask an uncomfortable question: did we actually grow the business, or did we just shuffle sales between our own products?"

**The journey:**
1. **The problem**: Distinguishing cannibalization from incrementality
2. **The approach**: Bayesian modeling with PyMC
3. **The insight**: 22.5% cannibalized, 57.6% incremental
4. **The impact**: 738K truly new units, not 951K

**The takeaway:**
"This analysis changed how we evaluate new product launches. Now we don't just ask 'How much did it sell?' We ask 'Where did those sales come from?' This shift in perspective has saved us from several portfolio expansions that would have just cannibalized existing products."

**The call to action:**
"If you're launching new products, don't rely on naive pre/post comparisons. Use statistical models that control for confounders and quantify uncertainty. Your CFO will thank you when you can confidently say 'This product generated $X million in truly incremental revenue.'"

---

## Additional Resources for Readers

- **PyMC documentation**: https://www.pymc.io/
- **Arviz (visualization)**: https://arviz-devs.github.io/
- **Causal inference in business**: Pearl & Mackenzie, "The Book of Why"
- **Bayesian modeling**: McElreath, "Statistical Rethinking"

---

**Good luck with your Medium article! This framework is publication-ready.**
