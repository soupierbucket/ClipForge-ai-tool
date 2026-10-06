# Student Successful Outcome Prediction

This project predicts `successful_outcome` for students in the UTS MDSI competition.

`successful_outcome = 1` means a successful outcome. `0` means an unsuccessful outcome. The competition evaluates predictions with ROC-AUC, so submissions contain probabilities between 0 and 1 rather than only `0` and `1` labels.

## Project structure

```text
ML/
├── Code.py                         # Stage 1: merge data to one row per student
├── feature_engineering.py          # Stage 2: add optional derived features
├── baseline_model.py               # Stage 3: train and evaluate logistic regression
├── compare_models_cv.py             # Stage 4: compare models with 5-fold validation
├── .venv/                          # Python virtual environment
└── uts-mdsi-competition-2026/
    ├── students.csv
    ├── student_profile_metrics.csv
    ├── student_attractions.csv
    ├── student_monthly_activity.csv
    ├── suburbs.csv
    ├── train.csv
    ├── combined_students.csv        # Created by Code.py
    ├── model_ready_students.csv      # Created by feature_engineering.py
    └── baseline_submission.csv       # Created by baseline_model.py
    └── cv_best_model_submission.csv  # Created by compare_models_cv.py
```

## Setup

Open a terminal in the `ML` folder and activate the virtual environment.

```bash
cd ~/Desktop/ML
source .venv/Scripts/activate
pip install pandas numpy scikit-learn
```

If activation is blocked, use the environment's Python executable directly:

```bash
.venv/Scripts/python.exe -m pip install pandas numpy scikit-learn
```

## Data sources

| File | Rows | Description |
|---|---:|---|
| `students.csv` | 25,000 | Student demographic, degree, work, accommodation, and lifestyle information |
| `student_profile_metrics.csv` | 25,000 | Academic engagement, career, commute, sleep, exercise, and screen-time measures |
| `suburbs.csv` | 15 | Location, rent, transport, safety, and community information |
| `student_attractions.csv` | 75,125 | Attraction visits; multiple rows per student |
| `student_monthly_activity.csv` | 281,790 | Monthly student activities; multiple rows per student |
| `train.csv` | 20,000 | Known training labels: `student_id` and `successful_outcome` |
| `sample_submission.csv` | 5,000 | Required submission layout |

There are 25,000 students in total: 20,000 labelled training students and 5,000 unlabelled students for prediction.

## Stage 1: merge the data

Run:

```bash
python Code.py
```

Models require a single row per student. `Code.py` starts from `students.csv`, then joins profile and suburb data. It aggregates the two tables that contain multiple records for each student before joining them.

### Attraction aggregation

For each student, it creates:

- number of attraction types visited
- total, average, and maximum attraction visits

### Monthly activity aggregation

For each student, it creates:

- months observed and activity duration
- total and average study hours
- total networking, club, job-application, beach, cultural-event, and internship activity
- average wellbeing score

### Merge checks

The script confirms every output row represents exactly one student. This prevents duplicate rows caused by an incorrect many-to-many join.

The output is `uts-mdsi-competition-2026/combined_students.csv`.

## Stage 2: feature engineering

Run:

```bash
python feature_engineering.py
```

This script keeps all original merged columns and adds optional student-level features.

### Activity trends

It calculates first-to-last changes in:

- study hours
- wellbeing score
- networking events
- job applications
- internship hours

### Ratios and per-month features

It adds per-month rates for study, networking, clubs, job applications, cultural activity, and internship hours. It also creates:

- academic participation average
- internship completion rate
- internship hours per completed internship
- career actions per month
- events per month
- work-to-study ratio
- commute minutes per kilometre
- weekly screen hours
- exercise sessions per week

### Attraction features

It adds attraction concentration and diversity measures. These are hypotheses to test, not assumed predictors of success. Keep them only if validation ROC-AUC improves.

### Log features

It creates log-transformed versions of several skewed count variables, while retaining their original values.

The output is `uts-mdsi-competition-2026/model_ready_students.csv`.

## Stage 3: baseline model

Run:

```bash
python baseline_model.py
```

The baseline script uses `model_ready_students.csv` when present; otherwise, it uses `combined_students.csv`.

### Features used

The model uses every column except:

- `student_id`, which is only an identifier
- `successful_outcome`, which is the value being predicted

This includes student profile, academic participation, career activity, work, commute, wellbeing, location, community, and aggregated activity features.

### Data preparation inside the model

Numeric columns are filled with the median from the training data. The model adds a flag indicating which numeric values were originally missing, then scales numeric values.

Categorical columns are filled with the most frequent training category and converted to one-hot encoded columns. For example, `transport_mode` becomes separate features such as `transport_mode_bus` and `transport_mode_train`.

These transformations are in the training pipeline, preventing validation data from influencing preprocessing.

### Validation

The 20,000 labelled students are split into:

```text
16,000 training rows
 4,000 validation rows
```

The split is stratified, so the proportion of successful outcomes remains similar in both sets.

### Model

The baseline uses logistic regression. It estimates a probability of success for each student by combining the available features.

### Evaluation: ROC-AUC

ROC-AUC measures ranking quality: successful students should receive higher predicted probabilities than unsuccessful students.

- `0.50`: random ranking
- `1.00`: perfect ranking

The first baseline run using `combined_students.csv` achieved:

```text
Single holdout validation ROC-AUC: 0.94450
```

After feature engineering, the same logistic-regression approach using
`model_ready_students.csv` achieved a single holdout ROC-AUC of `0.95266`.

After validation, the script retrains on all 20,000 labelled students and generates predictions for all 5,000 unlabelled students.

The output is `uts-mdsi-competition-2026/baseline_submission.csv`. This file
contains predictions from the most recent run of `baseline_model.py`.

## Stage 4: compare models with cross-validation

Run:

```bash
python compare_models_cv.py
```

This compares logistic regression, random forest, and extra trees using the
same five stratified folds. Each model's missing-value handling, scaling, and
category encoding are fitted inside each fold, avoiding preprocessing leakage.
The script reports all five ROC-AUC scores, their mean, and their standard
deviation. A higher mean indicates stronger ranking performance; a smaller
standard deviation indicates that scores varied less across folds.

Latest five-fold results:

| Model | Mean ROC-AUC | Standard deviation |
|---|---:|---:|
| Random forest | **0.96077** | 0.00189 |
| Logistic regression | 0.95518 | 0.00235 |
| Extra trees | 0.95497 | 0.00264 |

Random forest had the best mean score in this comparison. The script fits that
model using all labelled rows and writes its test probabilities to
`uts-mdsi-competition-2026/cv_best_model_submission.csv`. This is separate from
`baseline_submission.csv` so the two prediction files can be compared.

## Submission checks

Before uploading, verify that the chosen submission file has:

- exactly 5,000 rows
- columns named `student_id` and `successful_outcome`
- one unique `student_id` per row
- probabilities between `0` and `1`

## Next experiments

1. Compare candidate submissions and keep the file associated with the best
   validation performance.
2. Repeat model comparisons after any feature changes.
3. Try additional models or tune the random forest, while using the same folds
   for a fair comparison.
