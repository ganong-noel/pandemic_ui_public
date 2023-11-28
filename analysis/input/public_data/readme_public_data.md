# Sources for the public datasets in this directory

## Data from the Department of Labor (DOL)
ETA datasets can be found under the following link: [https://oui.doleta.gov/unemploy/DataDownloads.asp](https://oui.doleta.gov/unemploy/DataDownloads.asp)
- `ap902.csv`: ETA 902P 
- `elig_ui_reg_pua.csv`: this adds up the number of 'regular' UI payments (ETA 5159) and PUA payments (ETA 902P)
- `ui_receipt_benchmarks.xlsx`: sheets `month_final_payments` and `month_initial_claims` are from ETA 5159, and sheet `week_continued_claims` is from [https://oui.doleta.gov/unemploy/claims.asp](https://oui.doleta.gov/unemploy/claims.asp)

## Data from FRED
- `PAYEMS.xls`: [https://fred.stlouisfed.org/series/payems](https://fred.stlouisfed.org/series/payems)
- `PCE.xls`: [https://fred.stlouisfed.org/series/PCE](https://fred.stlouisfed.org/series/PCE)

## Data from the Bureau of Labor Statistics (BLS)
- `bls_payroll_emp_nonfarm_no_adj.xlsx`: select "Multi Screen" under "Employment, Hours, and Earnings - National" in section "Employment" of the following link [https://www.bls.gov/data/home.htm](https://www.bls.gov/data/home.htm) (alternatively, the series can be accessed here: [https://beta.bls.gov/dataViewer/view/timeseries/CEU0000000001](https://beta.bls.gov/dataViewer/view/timeseries/CEU0000000001))

## Data from other papers
- `literature_elasticities.csv` is a set of elasticities from Schmieder and Von Wachter 2016 

# Code in this directory
Stata code `decompose_pua.do` converts the data in `ap902.csv` to model input `ui_receipt_benchmarks.xlsx`. `monthly_exit_rates.dta` is an intermediate dataset saved by `decompose_pua.do`.
