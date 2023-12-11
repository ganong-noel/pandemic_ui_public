
import delimited "ap902.csv", varnames(1) clear
keep st rptdate c1 c4 c7
rename st state_abb
rename c1 pua_ic_total //total
rename c7 pua_ic_se //self-employed
rename c4 pua_cc
gen pua_ic_non_se = pua_ic_total - pua_ic_se
gen pct_se = pua_ic_se / pua_ic_total

gen month=mofd(date(rptdate, "MDY"))
format month %tm
order state_abb month rptdate pua_ic_total pua_ic_se pua_ic_non_se pct_se pua_cc

*keep if pct_se>.3 & pct_se<.95
replace pua_ic_se=. if pct_se<.5 | pct_se>.9
replace pua_ic_non_se=. if pct_se<.5 | pct_se>.9
collapse (sum) pua*, by(month)
*egen group=group(state_abb)
tsset month
* Want to still keep months that don't show up the raw monthly_exit_rates file (which only goes to May 2021)
merge 1:1 month using monthly_exit_rates.dta, keep(1 3)
replace f = f[_n-1] if missing(f)
replace mf = mf[_n-1] if missing(mf)
replace pua_cc=pua_cc/4
gen net_entry=pua_cc-l.pua_cc
gen exit=mf*pua_cc
replace exit=-1.25*net_entry if net_entry<-exit
gen exit_rate=exit/pua_cc
gen entry=net_entry+exit
gen share_allowed=entry/pua_ic_total

gen entry_self_employed=entry*pua_ic_se/(pua_ic_se+pua_ic_non_se)
gen stock_self_employed=entry_self_employed if _n==4
replace stock_self_employed=l.stock_self_employed*(1-exit_rate)+entry_self_employed if l.stock_self_employed!=.

*line stock_self_employed pua_cc month
gen ratio_stock=stock_self_employed/pua_cc
gen ratio_entry=entry_self_employed/entry
keep ratio* month
tempfile ratio
save `ratio'

import delimited "ap902.csv", varnames(1) clear
keep st rptdate c1 c4 c7
rename st state_abb
rename c1 pua_ic_total //total
rename c7 pua_ic_se //self-employed
rename c4 pua_cc
gen pua_ic_non_se = pua_ic_total - pua_ic_se
gen pct_se = pua_ic_se / pua_ic_total

gen month=mofd(date(rptdate, "MDY"))
format month %tm
order state_abb month rptdate pua_ic_total pua_ic_se pua_ic_non_se pct_se pua_cc


collapse (sum) pua*, by(month)
merge 1:1 month using `ratio', nogen

tsset month
merge 1:1 month using monthly_exit_rates.dta, keep(1 3)
replace f = f[_n-1] if missing(f)
replace mf = mf[_n-1] if missing(mf)
replace pua_cc=pua_cc/4
gen net_entry=pua_cc-l.pua_cc
gen exit=mf*pua_cc
replace exit=-1.25*net_entry if net_entry<-exit
gen exit_rate=exit/pua_cc
gen entry=net_entry+exit

gen continuing_claims_self_employed=ratio_stock*pua_cc
gen entry_self_employed=ratio_entry*entry

keep month continuing_claims_self_employed entry_self_employed
replace continuing_claims_self_employed=0 if month>=tm(2021m9)
replace entry_self_employed=0 if month>=tm(2021m9)
outsheet using "decompose_pua.csv", comma replace
