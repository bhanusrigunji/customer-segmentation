# Customer segmentation starter project

This project demonstrates customer analytics with the public Mall Customers sample: standardize annual income and spending score, compare candidate cluster counts, profile segments by age and gender, and translate profiles into testable marketing ideas.

## Quick start

### Open the interactive app

Double-click `outputs/customer_segmentation_app.html`, or open it from Command Prompt in the project folder:

```cmd
start "" outputs\customer_segmentation_app.html
```

The app opens in your browser with the 200-row sample ready. Choose a CSV to upload, adjust K, select at least two numeric clustering features, inspect the scatter plot and segment profiles, and export customer assignments as a CSV. The HTML is self-contained and does not require installing packages or running a server.

### Regenerate the analysis files

Run the Python analysis from the project folder with Python 3 and NumPy installed:

```cmd
python work\segment_customers.py
```

The script reads `outputs/Mall_Customers.csv` and recreates the analysis files in `outputs/`. The dataset is a public educational sample, not your own customer records. Segment names and suggested actions are interpretations to validate.

## Outputs

- `outputs/Mall_Customers.csv` — public sample data
- `outputs/segmented_customers.csv` — records with cluster labels
- `outputs/segment_profiles.csv` — customer counts and feature averages by segment
- `outputs/model_selection.json` — inertia and silhouette scores for k=2..7
- `outputs/segment_report.md` — findings and action ideas
- `outputs/segment_sizes.svg`, `outputs/segment_spend.svg` — charts
- `outputs/customer_segmentation_app.html` — interactive browser app

## Use real data

The current analysis clusters annual income and spending score. Age and gender describe the resulting groups but do not determine cluster assignment. For a richer behavioral analysis, replace the CSV and adapt the loader to include measures such as recency, purchase frequency, monetary value, channel activity, and promotion response. Exclude identifiers from clustering inputs, handle missing values and extreme values, then review candidate k diagnostics and profiles before using the results. The script has no external dependencies beyond NumPy.

## Data source

The sample CSV is the commonly used Mall Customers dataset. This project uses a copy from [GitHub](https://github.com/kennedykwangari/Mall-Customer-Segmentation-Data/blob/master/Mall_Customers.csv); the dataset is also listed on [Kaggle](https://www.kaggle.com/datasets/shwetabh123/mall-customers). Please check the original dataset's terms before redistributing it. The sample is educational and does not represent this project's author's business customers.

The project workflow was inspired by this [K-means customer segmentation video](https://www.youtube.com/watch?v=-LGwdrajMZ0).
