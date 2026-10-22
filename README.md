# Machine Learning Model Assessing the Temporal Stability of Food Insecurity Predictors

**Author:** Muhammad Mizanur Shuvra
**Course:** CIND 820 Big Data Analytics Project, Toronto Metropolitan University (Chang School)
**Current stage:** Milestone 2 (architecture and data audit), October 2026

## Project overview

This project assesses the temporal stability of food insecurity prediction models using Canadian survey data from 2018 to 2022, and examines whether adding more recent data improves their predictive performance.

It uses five annual Canadian Income Survey (CIS) Public Use Microdata Files (2018 to 2022). Each file is a separate cross-section of different households, so households are not followed over time.

## Research questions

1. **RQ1:** Which household characteristics are associated with food insecurity across Canadian survey periods from 2018 to 2022?
2. **RQ2:** How well do prediction models trained on pre-pandemic data generalize to subsequent survey periods?
3. **RQ3:** Does incorporating more recent data improve the out-of-time predictive performance of different modelling approaches?

## Data source and access

The data are published by Statistics Canada under the [Statistics Canada Open Licence](https://www.statcan.gc.ca/en/terms-conditions/open-licence):

- Canadian Income Survey, public-use microdata files, catalogue 72M0003X: <https://www150.statcan.gc.ca/n1/en/catalogue/72M0003X>
- Persistent link (DOI): <https://doi.org/10.25318/72m0003x-eng>

The five data files are large fixed-width text files (`CIS2018_PUMF.txt` to `CIS2022_PUMF.txt`), one row per person. They are **not stored in this repository**. Download them, together with the data dictionary for each year, from Statistics Canada. The dictionary for each year is needed because the column positions are read from it.

The Statistics Canada catalogue pages were temporarily unavailable on 21 October 2026. If a link does not open, try again later or search the Statistics Canada website for catalogue number 72M0003X.

## Analytical sample and outcome

- **Unit:** one row per household, taken from the record of the major income earner.
- **Households kept:** single-person households and households made up of one economic family. Households with two or more economic families are left out (3.0% to 3.4% of households each year), because the income variable would then describe only one of the families.
- **Outcome:** food insecurity status from the survey's household food security variable. It is coded 1 if the household is marginally, moderately or severely food insecure, and 0 if it is food secure.
- **Training and testing years:** train on 2018 and 2019; test on 2020 and on 2021 to 2022.

| Year | Households | Any food insecurity, unweighted | Any food insecurity, weighted |
|---|---|---|---|
| 2018 | 39,481 | 16.3% | 16.0% |
| 2019 | 30,712 | 14.7% | 15.7% |
| 2020 | 52,235 | 13.9% | 15.8% |
| 2021 | 38,015 | 15.9% | 17.6% |
| 2022 | 40,728 | 20.3% | 22.4% |
| **Total** | **201,171** | | |

## Predictors

| Variable | Meaning |
|---|---|
| `income_quintile` | After-tax economic-family income divided by the square root of household size, grouped into weighted fifths within each year (dollars of that year, no price adjustment). Zero and negative incomes are kept, in the lowest fifth. |
| `tenure` | Not owned, owned with a mortgage, or owned without a mortgage. |
| `has_children` | At least one person under 18 in the household. |
| `SEX` | Sex of the major income earner. |
| `edu_household` | Highest education among household members aged 18 and over. |
| `IMMST` | Immigrant status of the major income earner. |
| `MAJRI` | Major source of income of the major income earner, used as a **proxy** for the household's main income source. |
| `PROV` | Province. |
| `HHSIZE` | Household size. |

`SEX`, `IMMST` and `MAJRI` describe the major income earner, not the whole household. The other variables describe the household.

## Repository contents

| Item | What it is |
|---|---|
| `01_build_household_dataset.ipynb` | Reads the five annual files, checks them against the dictionaries, builds the household table and prints a PASS or FAIL line for each data check. |
| `02_eda.ipynb` | Exploratory data analysis: outcome by year, univariate and bivariate summaries, chi-square tests and Cramér's V. |
| `EDA_outputs/` | The tables and charts produced by notebook 02. |
| `ProcessedData/` | Created by notebook 01: `household_table.csv` and three small summary tables (missing values, sample flow and income cut-offs by year). If this folder is not in the repository, run notebook 01 to create it. |

## How to run the notebooks

1. Download the five CIS files and the data dictionaries from Statistics Canada (see above).
2. Create a folder named `CIND820_CapstoneProject` containing a subfolder `ProjectData`, and put the five `.txt` files in `ProjectData`. In Google Colab this folder sits in Google Drive; on your own computer it can be any local folder.
3. Open `01_build_household_dataset.ipynb`. In the SETTINGS block at the top, set `base_path` to your folder, then run all cells. Check that every data check prints PASS.
4. Open `02_eda.ipynb`, set `base_path` in the same way, and run all cells. The results are written to `EDA_outputs`.

The notebooks use Python with pandas, numpy, scipy and matplotlib, all of which are installed in Google Colab. Keep the original Statistics Canada files unchanged so that every step can be traced.

## Planned modelling (Milestones 3 and 4)

- **RQ1:** the same weighted logistic regression, fitted separately for each year (the baseline).
- **RQ2:** logistic regression, decision tree and random forest trained on 2018 to 2019, with model selection kept inside the training years, then applied unchanged to 2020 and to 2021 to 2022.
- **RQ3:** add newer years in sequence and test on the next period.

RQ2 and RQ3 will be run twice, with and without survey weights, and both sets of results will be reported side by side with no primary track.

Model fitting is not part of Milestone 2.

## Limits to keep in mind

- The files are repeated cross-sections of different households, so the results describe how associations and predictions hold across survey years, not how individual households change.
- The survey covers the ten provinces only, not the territories.
- The household's weight is the major income earner's person weight (`FWEIGHT`). The public file has no bootstrap weights, so weighted results report odds ratios, not p-values. The chi-square tests and Cramér's V in notebook 02 are unweighted.
- `SEX`, `IMMST` and `MAJRI` describe the earner, and `MAJRI` is a proxy.
- One predictor category, "no income" in `MAJRI`, has no households in the training years.
- `HHSIZE` and `has_children` overlap strongly (Cramér's V 0.78 in 2018 to 2019).
- Associations are not causes.

## Attribution and AI assistance

Source data: Statistics Canada, Canadian Income Survey (public-use microdata files), used under the Statistics Canada Open Licence.

Claude (Anthropic, Sonnet 5.5) drafted the code in notebooks 01 and 02 and the pipeline diagram, and wrote this README, based on an earlier README draft from ChatGPT. The author made the research decisions, ran and checked the notebooks, and is responsible for the work submitted. The report's Generative AI declaration gives the full account.
