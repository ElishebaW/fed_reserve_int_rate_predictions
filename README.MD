- Business Objective: Predict the future federal reserve interest rate in 2027 based on the current and previous interest rates
to make decisions on investments. 

- The current situation is unknown so this would be a use case to get informed using a machine learning model. 

- This is a supervised learning multi- regression problem using batch processing. 

- Will use root mean squared error as the metric to evaluate the model and it fits the business objective because this is a regression model and it will determine errors in predictions. 

- The Effective Federal Reserve Interest Rate is the average of the federal reserve interest rate over the past 12 months. 

- I assumed I had all the data I needed to make a prediction and train the model, but maybe I need more data. 

- These are the fields I will need to make a prediction:

    -   Inflation Rate (specifically the Personal Consumption Expenditures (PCE) index, which is the Fed's preferred gauge)( This is blank in some rows in the dataset)
    -   Unemployment Rate (a key component of the Fed's "dual mandate" for maximum employment)
    -   Real GDP (Percent Change) (used to assess economic growth and potential overheating)
    -   Effective Federal Funds Rate (provides the historical baseline for modeling future shifts)
    -   Federal Funds Target Rate (the primary policy tool being predicted)
    -   Federal Funds Upper and Lower Target (the boundaries set by the Fed for the effective rate)
    -   Time Period (Year, Month, and Day for time-series analysis# Model Tuning Notes (Interest Rate Project)

## Which tuning method is most appropriate here?
For this project, **HalvingRandomSearchCV** is the best default choice.

### Why
- You are using **time-series validation** (`TimeSeriesSplit`), so each candidate is relatively expensive.
- The Random Forest search space is multi-parameter (`max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `n_estimators`, etc.), where random sampling is more efficient than exhaustive grids.
- Successive halving allocates small resources to many configs first, then focuses compute on the most promising ones.

## Method ranking for this project
1. **HalvingRandomSearchCV**: best first pass (efficient + broad exploration)
2. **RandomizedSearchCV**: good baseline if you want simpler behavior
3. **HalvingGridSearchCV**: better than full grid, but grid itself is still less efficient in high-dimensional spaces
4. **GridSearchCV**: least efficient as an initial search over many params

## Time complexity intuition
In practice, search cost is often estimated as:

`total_cost ~= (#configs) * (#CV folds) * (cost of one model fit)`

### 1) GridSearchCV
- Tries all combinations.
- If parameter grid sizes are `k1, k2, ..., kd`, then `#configs = k1 * k2 * ... * kd`.
- Cost (roughly):

`O((prod_i ki) * F * C_fit(n, p))`

### 2) RandomizedSearchCV
- Tries a fixed number of sampled configs: `n_iter`.
- Cost (roughly):

`O(n_iter * F * C_fit(n, p))`

### 3) HalvingGridSearchCV
- Starts with many grid configs on low resources, keeps top fraction each round, increases resources each round.
- Cost (roughly):

`O(F * sum_r [N_r * C_fit(n_r, p)])`

where `N_r` decreases over rounds and `n_r` (resource per candidate) increases.

### 4) HalvingRandomSearchCV
- Same halving schedule, but with random initial configs instead of full grid.
- Same rough form as HalvingGrid, usually with better efficiency in larger spaces.

## Is "time complexity" a thing in ML?
Yes. In ML practice, people often discuss this as **computational cost** or **scaling behavior** rather than only strict asymptotic Big-O, because runtime depends on:
- algorithm,
- dataset size/features,
- CV strategy,
- and hardware.

## Suggested workflow for this project
1. Run **HalvingRandomSearchCV** with `TimeSeriesSplit` for broad search.
2. Take the best region and run a **small local GridSearchCV** for final refinement.
3. Evaluate final model on a strictly held-out chronological test split.





