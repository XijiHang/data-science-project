# PM2.5 Forecast Evaluation Beyond Average Error
Xiji Hang | Original analysis: 7 October 2026

Research Methods and Sensitivity Analysis

Research Question
Does selecting a next-day PM2.5 forecast by overall MAE hide weaker performance on high-concentration station-days?

Data Source and Coverage
Chen, S. (2017). Beijing Multi-Site Air Quality [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5RK5G. License: CC BY 4.0.
https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip
Access date: 7 October 2026. Raw archive SHA256: b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4
Original hourly records: 420,768. Sites: 12. PM2.5 missing hourly values: 8,739.
Archive contains a nested PRSA2017 zip with the twelve station CSVs. Small extra data.csv and test.csv files in the outer archive are not used.
Stations: Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng, Huairou, Nongzhanguan, Shunyi, Tiantan, Wanliu, Wanshouxigong
Complete station-date grid: 17,532; eligible adjacent-day forecast samples: 16,962; train: 12,030; test: 4,920.
Boundary samples deliberately excluded: 12 (one per site). No duplicated station-timestamps and no negative observed PM2.5 values.

Forecast Definition and Missing Values
At the close of day D, predict the mean PM2.5 for calendar day D+1. The target is an average over the coming day, not a prediction exactly 24 hours ahead at one instant.
Current and target PM2.5 days each need at least 18 of 24 observed hours. A low-coverage day is marked missing before any lag/target shift is calculated. No PM2.5 target is interpolated or imputed.
Daily means for the other numeric pollutant and weather variables are available only with >=18 measured hours; rain is the available-hour daily sum under the same coverage rule. Other missing features remain NaN and are handled natively by histogram boosting.
Lag features are generated on the complete station-calendar grid before filtering, so a lag means a true calendar lag, not the previous available row. The 7-day PM2.5 rolling mean uses at least five valid days and includes only day D and earlier.
Eligible test station-days are equally weighted. This is a site-day benchmark, not a population-weighted exposure estimate. Missingness can still change representativeness.

Temporal Validation
Training target dates: 2013-03-02 to 2015-12-31
Test origin dates: 2016-01-01 to 2017-02-27
Test target dates: 2016-01-02 to 2017-02-28
A training target precedes every test origin. Samples with origin 2015-12-31 and target 2016-01-01 are excluded. No random split is used.
Both boosting models are fitted once, then remain fixed throughout testing. New observations from a completed test day are legitimately available for the next forecast; models are not fitted to test outcomes. Test weather is never used from the day being predicted.

Predictors and Models
Numeric current-day means: PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, WSPM. Also current PM2.5 count and standard deviation, rain sum, PM2.5 lags at 1/2/7 days, trailing 7-day mean, target day-of-year sine/cosine and weekday. Station is categorical.
Target calendar information is known before forecasting. No future measured weather, target pollutant, target coverage count or test-derived normalization is used as a predictor.
Persistence predicts tomorrow equal to today's valid PM2.5 mean.
Boosting MAE uses HistGradientBoostingRegressor loss=absolute_error (conditional-median objective). Boosting MSE uses loss=squared_error (conditional-mean objective). The only configured difference is the loss.
Shared parameters: {"categorical_features": "from_dtype", "early_stopping": false, "l2_regularization": 10, "learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 15, "min_samples_leaf": 30, "random_state": 720407}
Early stopping is disabled. Hyperparameters are fixed and are not tuned on the test set. Negative boosting predictions are clipped at zero.
Software versions: {"numpy": "2.5.3", "pandas": "3.0.1", "scikit-learn": "1.9.1"}

High Concentrations and Forecast Alerts
High concentration uses the training target 90th percentile: 168.393840579710 ug/m3. This is an analysis threshold, not an official pollution category or health guideline.
Test high cases: 459 of 4920 (9.3293%). They occur on 54 distinct calendar dates among 424 test dates. Station-day counts are not independent episode counts.
A forecast alert means predicted PM2.5 reaches the SAME fixed concentration threshold. Recall=TP/(TP+FN), precision=TP/(TP+FP). MAE is the mean of absolute forecast errors; conditional tail MAE is computed only on observed high cases.

Results and 95 Percent Confidence Intervals
Persistence
  Overall MAE: 43.24 [37.27, 49.72] ug/m3
  High-case MAE: 90.47 [74.31, 107.39] ug/m3
  High-case recall: 52.51 [38.44, 64.00]%
  Alert precision: 51.50 [36.81, 63.64]%
  TP=241, FN=218, FP=227, TN=4234. Overall RMSE=65.124; high-case mean signed error=-62.581 ug/m3.
Boosting MAE
  Overall MAE: 35.90 [30.95, 41.29] ug/m3
  High-case MAE: 105.24 [85.73, 123.68] ug/m3
  High-case recall: 35.08 [23.25, 46.29]%
  Alert precision: 74.88 [61.93, 85.43]%
  TP=161, FN=298, FP=54, TN=4407. Overall RMSE=53.659; high-case mean signed error=-103.779 ug/m3.
Boosting MSE
  Overall MAE: 36.37 [31.86, 41.41] ug/m3
  High-case MAE: 90.95 [71.29, 109.61] ug/m3
  High-case recall: 44.01 [31.86, 54.52]%
  Alert precision: 67.11 [51.92, 79.55]%
  TP=202, FN=257, FP=99, TN=4362. Overall RMSE=52.357; high-case mean signed error=-84.727 ug/m3.

Paired Comparisons and Uncertainty
Circular calendar-day block bootstrap: 2,000 replicates, seven-day blocks, seed 720407. Starts are uniform over the full test calendar; blocks wrap around and are concatenated then truncated to 424 days. All retained stations from each sampled day are carried together.
All models use identical date draws. Every metric and the model difference are recalculated in each resample. Confidence limits are the 2.5th and 97.5th percentiles. The models and the training-derived threshold are not re-fitted in a resample.
These intervals describe approximate uncertainty in test metrics conditional on the fitted models and sampling assumptions. They do not include training-set uncertainty, new-site uncertainty, or a prediction interval for one future day. Seven-day blocks preserve short-range dependence but cannot guarantee that long-range or nonstationary dependence is fully represented.
Boosting MAE minus persistence:
  Overall MAE difference: -7.35 [-10.62, -4.04] ug/m3
  High-case MAE difference: 14.77 [2.05, 28.13] ug/m3
  Recall difference: -17.43 [-28.66, -5.01] percentage points
These three paired intervals exclude zero under this resampling scheme. This is more informative for paired comparisons than overlap of marginal intervals, and is not a universal model superiority test.

Sensitivity Analysis
Threshold sensitivity (same fitted models; training target quantiles define each cutoff):
  q85: cutoff 139.358 ug/m3; 668 high test cases.
    Persistence: high MAE 81.14, recall 51.95%, precision 51.10%.
    Boosting MAE: high MAE 87.07, recall 45.81%, precision 65.11%.
    Boosting MSE: high MAE 75.91, recall 57.19%, precision 66.20%.
  q90: cutoff 168.394 ug/m3; 459 high test cases.
    Persistence: high MAE 90.47, recall 52.51%, precision 51.50%.
    Boosting MAE: high MAE 105.24, recall 35.08%, precision 74.88%.
    Boosting MSE: high MAE 90.95, recall 44.01%, precision 67.11%.
  q95: cutoff 223.944 ug/m3; 248 high test cases.
    Persistence: high MAE 100.00, recall 39.11%, precision 38.96%.
    Boosting MAE: high MAE 142.55, recall 4.44%, precision 64.71%.
    Boosting MSE: high MAE 121.72, recall 17.34%, precision 62.32%.
Block-length sensitivity: 1,000 paired resamples per alternative length, with different fixed seeds.
  3 days: overall MAE difference CI [-10.77, -4.19]; high MAE difference CI [+2.25, +27.36]; recall difference CI [-27.87, -6.39].
  14 days: overall MAE difference CI [-10.88, -3.80]; high MAE difference CI [+0.93, +26.68]; recall difference CI [-29.70, -3.67].
Coverage sensitivity filters the existing test forecasts without refitting; this isolates evaluation-sample composition, not an alternative training pipeline.
  At least 20/24 observed hours in current AND target PM2.5 day: n=4869, high n=457.
    Persistence: overall MAE=43.38, high MAE=90.37, recall=52.74%.
    Boosting MAE: overall MAE=35.96, high MAE=104.96, recall=35.23%.
    Boosting MSE: overall MAE=36.40, high MAE=90.73, recall=44.20%.
  At least 24/24 observed hours in current AND target PM2.5 day: n=3729, high n=354.
    Persistence: overall MAE=44.77, high MAE=92.51, recall=55.08%.
    Boosting MAE: overall MAE=36.63, high MAE=108.05, recall=35.31%.
    Boosting MSE: overall MAE=37.07, high MAE=92.29, recall=46.33%.
The MAE-loss model's lower overall MAE and weaker tail detection than persistence persist in these checks. The MSE-loss comparison depends more on the threshold: at P85 it has higher recall than persistence; at P90 and P95 it has lower recall. The conclusion is specifically about using aggregate MAE alone, not that boosting must always underperform persistence.

Design Comparison
The aggregate-only comparison shows overall MAE, where boosting MAE is favourable. The final figure adds high-case MAE, recall and precision. Identical model rows link the panels, and a common error scale in panels A/B keeps the magnitude of tail errors visible.
The third model is a controlled loss-function comparison. It is not a claim that either fitted model represents the best achievable forecasting system. A loss function that gives larger errors more influence changes the practical trade-off in this specific test.
The shaded row guides readers to the main comparison; every marker has a number and each metric is defined. Marginal intervals and paired differences are labelled separately. No causal inference, regulatory exceedance definition, or current air-quality conclusion is made.

References
[1] Chen, S. (2017). Beijing Multi-Site Air Quality. UCI. doi:10.24432/C5RK5G. CC BY 4.0.
[2] scikit-learn. Cross-validation of time series data and Histogram-Based Gradient Boosting.
[3] arch documentation. Time-series bootstraps: circular blocks with end-to-start wrap-around.
https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data
https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting
https://bashtage.github.io/arch/devel/bootstrap/timeseries-bootstraps.html
