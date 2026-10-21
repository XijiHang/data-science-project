# Data source and permission

Chen, S. (2017). *Beijing Multi-Site Air Quality* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5RK5G.

- Dataset page: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
- Download: https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). UCI explicitly permits sharing and adaptation with appropriate credit. License checked on 21 October 2026.
- Original archive retrieved on 7 October 2026; SHA-256: `b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4`.
- Coverage: 12 Beijing monitoring stations; 1 March 2013 to 28 February 2017; 420,768 hourly rows.
- Air quality measurements originate from the Beijing Municipal Environmental Monitoring Center. Weather observations are matched to nearby China Meteorological Administration stations, as described by UCI.

## Changes made

Hourly records are aggregated into daily station summaries. Daily inputs with fewer than 18 measured hours are marked missing. Calendar lags, a trailing mean and known target-day calendar variables are added. The next day's valid PM2.5 mean becomes the target. Models generate forecasts and aggregate evaluation statistics. The figures and numerical summaries in this repository are derived outputs, not original UCI figures. Credit the source and identify these transformations when reusing derived material.

The data describe environmental monitoring stations, not individual people. Raw data are downloaded from UCI rather than copied into this repository. No Moodle teaching material, account information, passwords or restricted datasets are included.
