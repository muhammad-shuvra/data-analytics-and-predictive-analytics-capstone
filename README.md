# Machine Learning Model Assessing the Temporal Stability of Food Insecurity Predictors

**Author:** Muhammad Mizanur Shuvra  
**Course:** CIND820 Capstone, Toronto Metropolitan University  
**Current stage:** Milestone 3 — initial results and code (a working baseline model with validation by year)

## Project overview

This project examines whether models developed using earlier Canadian survey data remain useful for predicting household food insecurity in later periods. It also examines whether incorporating newer data improves subsequent predictive performance.

The analysis uses Canadian Income Survey (CIS) Public Use Microdata Files for the 2018–2022 reference years.

## Research questions

1. Which household characteristics are associated with food insecurity across the study years?
2. How well do models trained using the 2018–2019 files generalize to later periods?
3. Does incorporating newer data improve out-of-time predictive performance?

## Where the project stands

| Milestone | What it covers | Files |
|---|---|---|
| 1. Project design | Question, data and approach | report |
| 2. Architecture and data audit | Reading the raw files, building the household dataset, exploring the data | notebooks 00, 01, 02 |
| 3. Initial results and code | First working model for RQ2: a logistic regression fitted on 2018, checked on 2019, refitted on 2018–2019 and tested on 2020, 2021 and 2022 | notebook 003 and `m3_helpers.py` |
| 4. Final analysis (planned) | Further models, RQ1 and RQ3, stability and model comparison | not started |

## Data source and access

The source data and accompanying dictionaries are published by Statistics Canada:

- [Canadian Income Survey public-use microdata — Catalogue 72M0003X](https://www150.statcan.gc.ca/n1/en/catalogue/72M0003X)
- [Persistent product DOI](https://doi.org/10.25318/72m0003x-eng)

Download the 2018–2022 files and the corresponding data dictionaries. If a link is temporarily unavailable, search the Statistics Canada website for catalogue number **72M0003X**.

The source files contain structured fixed-width text. Their layouts must be interpreted using the dictionary for the corresponding year.

The prepared dataset, `household_table.csv` (one row per household, built by notebook 01), is not stored in this repository. It can be downloaded here: [household_table.csv](https://drive.google.com/file/d/1hKqMwiAFFjs3ku_oaWYMyZQffNIIUMCi/view?usp=sharing) (anyone with the link can view it). The same link is saved in `data/data_link.md`.

## Analytical sample and outcome

The prepared dataset contains **201,171 household observations** across five survey cycles.

One household record is selected using the major-income-earner indicator. The analysis retains single-person households and households containing one economic family.

The binary outcome (`food_insecure`, created from the survey variable `FSCHHLDM`) distinguishes food-secure households from households reporting marginal, moderate or severe food insecurity. This analysis does not link individual households across years.

## Milestone 2 work

The completed data work covers:

- Reading and checking the annual fixed-width files.
- Harmonizing variables and constructing the household dataset.
- Examining special response codes and missingness.
- Summarizing the outcome and predictor distributions.
- Comparing descriptive food-insecurity rates across groups.
- Examining categorical associations using Cramér’s V.

The report and notebooks document the preparation decisions and exploratory results.

## Milestone 3 work

Notebook 003 uses the prepared household table and covers:

- Checking the household counts against the Milestone 2 report.
- Descriptive statistics, histograms and boxplots for adjusted income and household size (2018–2019).
- A split by year, never at random: fit on 2018, validate on 2019, refit on 2018–2019, test once each on 2020, 2021 and 2022.
- A logistic regression on nine predictors, each treated as a category, with missing values kept as their own category. Two versions run side by side: one counts every household equally, one weights households by FWEIGHT.
- **Classification threshold selection.** The probability cut-off was chosen using the 2019 validation data only. Thresholds between 0.05 and 0.95 were evaluated, and the cut-off that maximized the F1 score was selected because it provided a balanced trade-off between precision and recall. The selected cut-offs (0.23 for the unweighted track and 0.25 for the weighted track) were frozen before any 2020-2022 test data were examined and were applied unchanged to all test years. This approach prevents information from future survey cycles from influencing model development and provides a cleaner assessment of how well a model developed on earlier years generalizes to later years.
- Confusion matrices, ROC-AUC and PR-AUC.
- A check of the overlap between household size and presence of children (2018 fit, 2019 validation).

The reusable functions are in `code/m3_helpers.py`.

## Planned modelling

- **RQ1:** Fit comparable weighted logistic regressions separately by year.
- **RQ2:** Train logistic regression, decision tree and random forest models using 2018–2019 data, then evaluate the unchanged models on 2020–2022. Milestone 3 covers the logistic regression only.
- **RQ3:** Add newer completed years sequentially and compare updated and fixed models on the same subsequent held-out year.

Cross-validation will be restricted to the training data. Confusion matrices will help describe classification performance on evaluation samples.

The decision tree, the random forest, RQ1 and RQ3 are planned work for Milestone 4.

## Folder structure

```
README.md
notebooks/         00_data_check.ipynb
                   01_build_household_dataset.ipynb
                   02_eda.ipynb
                   003_initial_results_logistic_regression_simple.ipynb
compiled/          HTML and PDF versions of notebook 003
code/              m3_helpers.py
data/              README.md (data notes and download link for household_table.csv)
ProcessedData/     output of notebook 01
EDA_outputs/       tables and figures from notebook 02
docs/              setup guides: 1_get_the_data.md, 2_open_in_colab.md, 3_check_variable_counts.md, FAQ.md
```

## Running the notebooks

The notebooks run in Google Colab and read and write files in Google Drive, in a folder called `CIND820_CapstoneProject`:

```
CIND820_CapstoneProject/
  ProjectData/      the five CIS PUMF text files (CIS2018_PUMF.txt ... CIS2022_PUMF.txt)
  ProcessedData/    household_table.csv (made by notebook 01)
  EDA_outputs/      tables and figures (made by notebook 02)
  code/             m3_helpers.py
  M3_outputs/       made by notebook 003
```

1. Download the required CIS files and dictionaries, and copy the files into the Drive folder as shown.
2. Open each notebook in Google Colab and run it from the top. Each notebook starts by mounting Google Drive, so click Allow when asked.
3. Run the notebooks in this order: 00, 01, 02, then 003. To run only notebook 003, download `household_table.csv` from the link above, put it in `ProcessedData/`, and put `m3_helpers.py` in `code/`.
4. Notebook 003 saves its tables and figures in `M3_outputs`. Review the saved outputs alongside the report.

Keep the original source files unchanged so that preparation steps remain traceable.

Software: Google Colab with pandas, numpy, matplotlib and scikit-learn. Notebook 003, Step 1, prints the versions. In the run submitted for Milestone 3 these were pandas 2.2.3 and scikit-learn 1.6.1.

## Interpretation

Weighted descriptive results use FWEIGHT from the selected major-income-earner record as a working household weight. Estimates should be interpreted within the project’s sample restrictions and weighting approach.

The annual files are repeated cross-sectional surveys. Associations do not establish causation, and exploratory findings do not demonstrate that predictive performance has changed over time.

## Attribution and AI assistance

Source data: Statistics Canada, Canadian Income Survey.

Data use follows the [Statistics Canada Open Licence](https://www.statcan.gc.ca/en/terms-conditions/open-licence).

Claude prepared the final set of code and the helper file, and assisted with troubleshooting and with preparing the tables and figures. ChatGPT prepared the initial version of the code and was consulted for initial planning and literature organization. The report's Generative AI declaration provides further details. The author is responsible for all final decisions, the writing in his own words, code execution and verification, and the submitted work.
