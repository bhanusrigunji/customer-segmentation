# Mall customer segmentation results

**Dataset:** Mall Customers public sample, 200 customer records. **Method:** z-score standardization and K-means with k=5, using annual income and spending score. Age and gender are used to describe the resulting groups, not to form the clusters. Candidate k=2–7 diagnostics are in `model_selection.json`. This dataset does not include purchase history, so the spending score is a provided index, not actual transaction data.

| Segment | Customers | Share | Avg age | Avg income (k$) | Avg spending score | Female |
|---|---:|---:|---:|---:|---:|---:|
| Budget-conscious | 23 | 11.5% | 45.2 | 26.3 | 20.9 | 60.9% |
| Aspiring spenders | 22 | 11.0% | 25.3 | 25.7 | 79.4 | 59.1% |
| Mid-market regulars | 80 | 40.0% | 42.9 | 55.1 | 49.7 | 58.8% |
| Careful high earners | 36 | 18.0% | 40.7 | 87.8 | 17.6 | 47.2% |
| High-value enthusiasts | 39 | 19.5% | 32.7 | 86.5 | 82.1 | 53.8% |

## Possible actions to validate

- **Budget-conscious (11.5%):** Test affordable bundles and value-led offers; measure margin as well as response.
- **Aspiring spenders (11.0%):** Test relevant cross-sells and accessible premium options; avoid assuming high spending is recurring.
- **Mid-market regulars (40.0%):** Use broad assortment and loyalty tests; look for opportunities to increase engagement.
- **Careful high earners (18.0%):** Test personalized recommendations and premium discovery without relying on discounting.
- **High-value enthusiasts (19.5%):** Prioritize curated launches and loyalty recognition; monitor retention and basket mix.

## Files

- `Mall_Customers.csv`: downloaded public sample input.
- `segmented_customers.csv`: input records with assigned segment.
- `segment_profiles.csv`: segment sizes and average characteristics.
- `model_selection.json`: inertia and silhouette diagnostics for candidate k.
- `segment_sizes.svg`, `segment_spend.svg`: charts.
