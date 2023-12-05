# ui_benchmarking_new.R
# Author: Katie Zhang
# Date: 2020-02-01
# This script benchmarks unemployment insurance (UI) claims and payments data
# from the Department of Labor (DOL) Employment & Training Administration (ETA)
# against Chase internal data.
# Benchmarks produced include: time series plots of DOL claims against
# Chase weeks paid, hexbin map of ratio of weeks paid, state heterogeneity in
# increase from 2019 to 2020, comparison of weekly benefits level.

out_path_benchmarking <- str_c(out_path, "/ui_benchmarking/")

######## ~ SECTION: READ IN DATA ##############################################
# Set up fips
fips_codes <- maps::state.fips %>%
  select(state_fips = fips,
         state_name = polyname,
         state_abb  = abb) %>%
  mutate(state_name = sub(":.*", "", state_name)) %>%
  distinct() %>%
  bind_rows(tibble(state_abb = c("HI", "AK"),
                   state_name = c("hawaii", "alaska"),
                   state_fips = c(15, 02))) %>%
  arrange(state_fips) %>%
  mutate(state_name = if_else(state_abb == "DC",
                              "District of Columbia",
                              gsub("(?<=\\b)([a-z])", "\\U\\1", tolower(state_name), perl=TRUE)))

#### Read in Chase weekly summary data ####
chase_bmark <-
  read_xls(job_find_file,
           sheet = "weekly_summary_natl_export") %>%
  mutate(puc_week = week_start_date > ymd("2020-04-05") &
           week_start_date < ymd("2020-08-01"),
         week_end_date = ceiling_date(ymd(week_start_date), unit = "week") - 1)

#### Read in Chase state spells data ####
chase_state_spells <-
  read_xls(job_find_file,
           sheet = "n_wks_paid_state") %>%
  rename(state_abb = cust_state,
         spell_active = n_weeks_w_active_spell) %>%
  mutate(period = case_when(
    time == "Jan & Feb 2020" ~ "precov",
    time %in% c("2020-03-01", "2020-03-08") ~ "precov",
    time == "2019" ~ "2019",
    grepl("-05-", time) ~ "may",
    TRUE ~ "postcov"),
    time = if_else(
      time == "Jan & Feb 2020",
      "janfeb2020",
      time),
    spell_active = if_else(
      # IN's n claims in base period adjusted - only Feb shows up
      state_abb == "IN" & time == "janfeb2020",
      spell_active * 2,
      spell_active))

chase_states <- unique(chase_state_spells$state_abb)

#### Read in Chase weekly benefit data ####
chase_bmark_benefit_raw <-
  read_xls(job_find_file,
           sheet = "typical_benefits_state") %>%
  mutate(payments_month = round(n_weekly / n_monthly, digits = 0),
         frequency = case_when(
           payments_month == 2 ~ "Fortnightly",
           payments_month == 4 ~ "Weekly",
           TRUE ~ "Weekly"),
         has_monthly = !is.na(n_monthly)) %>%
  rename(state_abb = cust_state,
         median_weekly_2019 = pseudomedian_weekly_2019,
         mean_weekly_2020 = mean_weekly,
         n_weekly_2020 = n_weekly)

#### Read in Chase counts (all vs primary) data ####
hero_universe_df <- 
  read_xls(job_find_file,
           sheet = "hero_universe_df") %>%
  mutate(week_start_date = floor_date(as.Date(week_start_date), "weeks")) %>%
  select(c(week_start_date, spell_active, key)) 

chase_counts <- 
  hero_universe_df %>% 
  filter(key == "Drop Account Activity Screens") %>% 
  rename(ct_cust_w_ui = spell_active) %>% 
  select(-key) %>% 
  inner_join(hero_universe_df %>% 
               filter(key == "Baseline") %>% 
               rename(ct_core_cust_w_ui = spell_active) %>% 
               select(-key), 
             by = "week_start_date") %>% 
  mutate(ratio = ct_cust_w_ui/ct_core_cust_w_ui)

#### Read in DOL weekly claims data ####
## Weekly state UI program claims
state_ui <- read_csv("analysis/input/ar539.csv") %>%
  transmute(state_abb = st,
            week = mdy(c2),
            state_ic = c3 + c4 + c5 + c6 + c7,
            state_cc = c8 + c9 + c10 + c11 + c12 + c13) %>%
  filter(!state_abb %in% c("PR", "VI")) %>%
  group_by(state_abb) %>%
  mutate(state_ic = lag(state_ic)) %>% # Rematch the weeks of ic vs cc
  full_join(fips_codes, by = "state_abb") %>%
  relocate(state_name, state_abb, state_fips) %>%
  arrange(state_fips) %>%
  filter(year(week) >= 2015)

## Pandemic claims
pandemic_ui <- read_xlsx("analysis/input/weekly_pandemic_claims.xlsx") %>%
  transmute(state_abb = State,
            week      = ymd(`Reflect Date`),
            pua_ic    = `PUA IC`,
            pua_cc    = `PUA CC`,
            peuc_cc   = `PEUC CC`) %>%
  filter(!state_abb %in% c("PR", "VI")) %>%
  group_by(state_abb) %>%
  mutate(pua_ic = lag(pua_ic)) %>% # Rematch the weeks of ic vs cc
  full_join(fips_codes, by = "state_abb") %>%
  relocate(state_name, state_abb, state_fips)

# Join both together
all_ui <- state_ui %>%
  full_join(pandemic_ui,
            by = c("state_name", "state_abb", "state_fips", "week")) %>%
  mutate_if(is.numeric, ~replace(., is.na(.), 0)) %>%
  mutate(dol_ic = state_ic + pua_ic,
         dol_cc = state_cc + pua_cc + peuc_cc) %>%
  filter(between(week, ymd("2019-01-01"), ymd("2020-10-31")))

#### Read in monthly data ####
# Regular Program
regular_ui <- read_csv("analysis/input/ar5159.csv") %>%
  transmute(
    state_abb = st,
    month     = floor_date(mdy(rptdate), "month"),
    rptdate   = mdy(rptdate),
    reg_weeks_paid  = c38 + c41,
    reg_amount_paid = c45 + c48,
    reg_ic          = c1 + c97 + c7 + c98 + c13 + c99,
    reg_cc          = c21 + c22 + c27 + c28 + c33 + c34,
    reg_first_payments = c51 + c54 + c55,
    reg_final_payments = c56 + c57 + c58
  ) %>%
  filter(!(state_abb %in% c("PR", "VI")))

# Extended Benefits
extended_ui <- read_csv("analysis/input/ae5159.csv") %>%
  transmute(
    state_abb = st,
    month   = floor_date(mdy(rptdate), "month"),
    rptdate = mdy(rptdate),
    eb_weeks_paid  = c29 + c31,
    eb_amount_paid = c35 + c37,
    eb_ic   = c1 + c46 + c2 + c49 + c4 + c47 + c5 + c50 + c7 + c48 + c8 + c51,
    eb_cc   = c12 + c13 + c18 + c19 + c24 + c25,
    eb_first_payments = c40 + c41 + c42,
    eb_final_payments = c43 + c44 + c45
  ) %>%
  filter(!(state_abb %in% c("PR", "VI")))

# PEUC
peuc_ui <- read_csv("analysis/input/ap5159.csv") %>%
  transmute(
    state_abb = st,
    month   = floor_date(mdy(rptdate), "month"),
    rptdate = mdy(rptdate),
    peuc_weeks_paid  = c29 + c31,
    peuc_amount_paid = c35 + c37,
    peuc_ic = c1 + c46 + c2 + c49 + c4 + c47 + c5 + c50 + c7 + c48 + c8 + c51,
    peuc_cc = c12 + c13 + c18 + c19 + c24 + c25,
    peuc_first_payments = c40 + c41 + c42,
    peuc_final_payments = c43 + c44 + c45
  ) %>%
  filter(!(state_abb %in% c("PR", "VI")))

# Combine all 3
dol_payments <- regular_ui %>%
  full_join(extended_ui) %>%
  full_join(peuc_ui) %>%
  filter(month < ymd("2020-11-01")) %>% #not all states have reported
  mutate_if(is.numeric, ~replace(., is.na(.), 0)) %>%
  mutate(month_weeks = difftime(rptdate, month, units = "weeks"),
         all_weeks_paid =
           reg_weeks_paid + eb_weeks_paid + peuc_weeks_paid,
         all_amount_paid =
           reg_amount_paid + eb_amount_paid + peuc_amount_paid,
         all_ic =
           reg_ic + eb_ic + peuc_ic,
         all_cc =
           reg_cc + eb_cc + peuc_cc,
         all_claims = all_ic + all_cc,
         all_first_payments =
           reg_first_payments + eb_first_payments + peuc_first_payments,
         all_final_payments =
           reg_final_payments + eb_final_payments + peuc_final_payments)

######## ~ SECTION: PLOTS ######################################################

#### Combine Chase counts with DOL data
benchmark_counts <- chase_counts %>%
  left_join(all_ui %>%
              # Lag DOL a week to reflect time lapse before payment
              mutate(week_start_date = floor_date(week, "weeks") + 7) %>%
              group_by(week_start_date) %>%
              summarise(dol_ic = sum(dol_ic, na.rm = TRUE),
                        dol_cc = sum(dol_cc, na.rm = TRUE))) %>%
  filter(!is.na(dol_cc)) %>%
  mutate(period = if_else(
    week_start_date >= ymd("2020-01-01") & week_start_date < ymd("2020-03-15"),
    "pre_covid",
    "other")) %>%
  group_by(period) %>%
  mutate(pc_allcust = mean(ct_cust_w_ui),
         pc_primary = mean(ct_core_cust_w_ui),
         pc_dolcc   = mean(dol_cc)) %>%
  ungroup() %>%
  mutate(pc_allcust = pc_allcust[which(week_start_date == ymd("2020-03-01"))],
         pc_primary = pc_primary[which(week_start_date == ymd("2020-03-01"))],
         pc_dolcc = pc_dolcc[which(week_start_date == ymd("2020-03-01"))]) %>%
  transmute(week_start_date,
            allcust_lvls = ct_cust_w_ui,
            allcust_norm = allcust_lvls / pc_allcust,
            primary_lvls = ct_core_cust_w_ui,
            primary_norm = primary_lvls / pc_primary,
            dolcc_lvls   = dol_cc,
            dolcc_norm   = dol_cc / pc_dolcc,
            dolcc_rescaled = dol_cc / plyr::round_any(mean(dol_cc / allcust_lvls, na.rm = TRUE), 10)) %>% #nolint
  pivot_longer(-week_start_date,
               names_to = c("source", "scale"),
               names_sep = "_",
               values_to = "value") %>%
  mutate(source = factor(source,
                         levels = c("allcust", "primary", "dolcc")),
         scale  = factor(scale,
                         levels = c("lvls", "norm", "rescaled")),
         source_scale = str_c(source, "_", scale))

#### Combine chase state spells with DOL (for hexmap)
chase_state_spells_periods <- chase_state_spells %>%
  select(-time) %>%
  filter(period != "postcov") %>%
  group_by(period, state_abb) %>%
  summarise(spell_active = sum(spell_active)) %>%
  pivot_wider(names_from = period,
              values_from = spell_active,
              names_prefix = "chase_")

dol_long <- all_ui %>%
  filter(state_abb %in% chase_states) %>%
  mutate(period = case_when(
    year(week) == 2019 ~ "2019",
    week >= ymd("2020-01-01") &
      week < ymd("2020-03-15") ~ "precov",
    week >= ymd("2020-05-01") &
      week < ymd("2020-06-01") ~ "may",
    TRUE ~ "drop")) %>%
  filter(period != "drop") %>%
  group_by(state_abb, state_name, period) %>%
  summarise(dol_cc = sum(dol_cc)) %>%
  ungroup() %>%
  pivot_wider(names_from = period,
              values_from = dol_cc,
              names_prefix = "dol_")

chase_dol_states <- dol_long %>%
  left_join(chase_state_spells_periods) %>%
  mutate(ratio_weeks_paid_2019 = chase_2019 / dol_2019)

#### Time series plots ####
# Plot with one axis for Chase
date_breaks <- ymd(c(20200201,
                     20200401,
                     20200601,
                     20200801,
                     20201001))
date_labels <- format(date_breaks, "%b")
date_labels[2] <- str_c(date_labels[2], "\n(Begin supplement)")
date_labels[4] <- str_c(date_labels[4], "\n(End supplement)")

## All vs primary
# Normalised relative to pre-covid
maxvalnorm <- benchmark_counts %>%
  filter(scale == "norm") %>%
  filter(value == max(value)) %>%
  select(value) %>%
  pull() %>%
  plyr::round_any(., 5, f = ceiling)

benchmark_counts %>%
  filter(scale == "norm",
         week_start_date >= ymd("2020-01-01"),
         week_start_date < ymd("2020-09-01")) %>%
  ggplot() +
  aes(x = ymd(week_start_date), y = value, colour = source) +
  geom_line() +
  fte_theme() +
  scale_x_date(name = NULL, date_breaks = "months", date_labels = "%b") +
  scale_y_continuous(limits = c(0, maxvalnorm)) +
  scale_colour_manual(labels = c("JPMCI (all)",
                                 "JPMCI (met account activity screens)",
                                 "Department of Labor"),
                      values = c(uieip_palette[1:3])) +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 12)) +
  labs(y = NULL,
       subtitle = "Ratio of JPMCI payments and DOL claims to average pre-Covid 2020 levels") #nolint

ggsave(str_c(out_path_benchmarking, "diagnostic_levels_norm.png"),
       width = 8, height = 4.5)

#### Hexmap - in sample ####
spdf <- geojson_read("analysis/input/us_states_hexgrid.geojson",
                     what = "sp")

spending_jobfinding_expiry_onset <- 
  c("NY", "IN", "WA", "GA", "NJ", "OH", "IL", "CA")
spending_jobfinding_expiry <- c("MI", "OR")
spending_jobfinding_onset <- c("TX", "WI", "LA", "CT")
not_in_sample <- c("MD", "HI", "KY", "MT", "NV", "OK")

spending_jobfinding_any <- c(spending_jobfinding_expiry_onset,
                             spending_jobfinding_expiry,
                             spending_jobfinding_onset)
spending_only_complement <- c(spending_jobfinding_any,
                              not_in_sample)
spending_only <- setdiff(fips_codes$state_abb, spending_only_complement)

spdf_data <- spdf@data %>%
  mutate(google_name = gsub(" \\(United States\\)", "", google_name)) %>%
  rename(state_abb = iso3166_2)

region_id <- data.frame(state_name = spdf_data$google_name) %>%
  mutate(id = 1:n())

spdf_fortified <- tidy(spdf) %>%
  mutate(id = gsub(" \\(United States\\)", "", id))

geo_centres <- cbind.data.frame(
  data.frame(gCentroid(spdf, byid = TRUE),
             id = spdf_data$state_abb))

spdf_in_sample <- spdf_fortified %>%
  mutate(id = as.numeric(id)) %>%
  right_join(region_id, by="id") %>%
  left_join(fips_codes, by="state_name") %>%
  select(-id) %>%
  rename(id=state_name)%>%
  mutate(sample_status = 
           case_when(state_abb %in% spending_jobfinding_expiry_onset 
                     ~ "spending_expiry_onset",
                     state_abb %in% spending_jobfinding_expiry 
                     ~ "spending_expiry",
                     state_abb %in% spending_jobfinding_onset 
                     ~ "spending_onset",
                     state_abb %in% spending_only ~"spending",
                     TRUE ~ "not_in_sample"))

ggplot() +
  geom_polygon(data = spdf_in_sample,
               aes(fill = sample_status,
                   x = long,
                   y = lat,
                   group = group),
               color = "white") +
  geom_text(data = geo_centres,
            aes(x = x, y = y, label = id),
            color = "white") +
  scale_fill_manual(
    values = 
      c("spending_expiry_onset" = "#004577",
        "spending_expiry" = "#2575ae",
        "spending_onset" = "#00a1df",
        "spending" = "#91c4c5",
        "not_in_sample" = "grey80"),
    labels = 
      c("spending_expiry_onset" = "Spending + job finding (expiration & onset)",
        "spending_expiry" = "Spending + job finding (expiration)",
        "spending_onset" = "Spending + job finding (onset)",
        "spending" = "Spending",
        "not_in_sample" = "Not in sample")) +
  theme_void() +
  theme(plot.background = element_rect(fill = "white", color = "white")) +
  coord_map() +
  theme(legend.title = element_blank())

ggsave(filename = str_c(out_path_benchmarking, "hexmap_jpmci_sample.png"),
       width = 8, height = 4.5, units = "in")

#### Capturing state heterogeneity in increases ####
## Ratio of pre-covid 2020 to May average
state_claim_increases <- chase_dol_states %>%
  select(-ratio_weeks_paid_2019) %>%
  pivot_longer(cols = c(-"state_abb", -"state_name"),
               names_to = c("series", "period"),
               names_sep = "_",
               values_to = "claims") %>%
  mutate(wkly_claims =
           case_when(period == "may" ~
                       claims / as.numeric(difftime(ymd("2020-05-31"),
                                                    ymd("2020-05-01"),
                                                    units = "weeks")),
                     period == "precov" ~
                       claims / as.numeric(difftime(ymd("2020-03-15"),
                                                    ymd("2020-01-01"),
                                                    units = "weeks")),
                     TRUE ~
                       claims / as.numeric(difftime(ymd("2020-01-01"),
                                                    ymd("2019-01-01"),
                                                    units = "weeks"))),
         wkly_claims = round(wkly_claims, digits = 0)) %>%
  rename(total_claims = claims) %>%
  pivot_wider(names_from = c("series", "period"),
              values_from = c("total_claims", "wkly_claims")) %>%
  mutate(dol_inc = wkly_claims_dol_may / wkly_claims_dol_precov,
         chase_inc = wkly_claims_chase_may / wkly_claims_chase_precov)

## Scatter - Include 45 degree line & don't drop FL
state_claim_increases %>%
  filter(total_claims_chase_2019 > 300) %>% #dropping small states, see #379
  ggplot() +
  aes(x = dol_inc,
      y = chase_inc,
      #colour = state_abb,
      label = state_abb) +
  geom_point(color = "#004577") +
  geom_text_repel(max.overlaps = Inf) + 
  scale_x_continuous(labels=function(x) paste0(x,"%")) +
  scale_y_continuous(labels=function(x) paste0(x,"%")) +
  fte_theme() +
  labs(x = "Department of Labor increase in continued claims",
       y = "",
       subtitle = "JPMCI increase in payments") +
  expand_limits(x = 0, y = 0) +
  geom_abline(linetype = "longdash")

ggsave(str_c(out_path_benchmarking, "state_hetero_inc_scatter.png"),
       width = 8, height = 4.5, units = "in")

#### JPMCI selecting on high/low wages ####
## Chase
chase_bmark_benefit <- chase_bmark_benefit_raw %>%
  select(state_abb, contains("weekly"), frequency, has_monthly) %>%
  pivot_longer(cols = contains("weekly"),
               names_to = c("series", "weekly", "year"),
               names_sep = "_",
               values_to = "chase_value") %>%
  mutate(chase_value = if_else(frequency == "Fortnightly" & series != "n",
                               chase_value / 2,
                               chase_value)) %>%
  select(-weekly)

## DOL - note this is from monthly values
dol_wkly_benefit <-
  dol_payments %>%
  filter(year(month) >= 2019) %>%
  transmute(state_abb, month,
            dol_weeks_paid   = all_weeks_paid,
            dol_amount_paid  = all_amount_paid,
            period = case_when(
              year(month) == 2019 ~ "2019",
              month == ymd("2020-02-01") ~ "2020",
              TRUE ~ "other"
            )) %>%
  filter(period != "other") %>%
  group_by(state_abb, year = period) %>%
  summarise(dol_amount_paid = sum(dol_amount_paid),
            n = sum(dol_weeks_paid)) %>%
  ungroup() %>%
  mutate(mean = dol_amount_paid / n,
         median = mean) %>%
  select(-dol_amount_paid) %>%
  pivot_longer(cols = c("mean", "median", "n"),
               names_to = "series",
               values_to = "dol_value")

## Both wkly benefit
compare_wkly_benefit <-
  dol_wkly_benefit %>%
  full_join(chase_bmark_benefit)

#### Scatterplot comparing weekly benefits ####
# Monthly vs not monthly
vector_small_states <- compare_wkly_benefit %>%
  filter(year == 2019) %>%
  filter(series == "n") %>%
  filter(chase_value < 300) %>%
  select(state_abb) %>%
  pull()

compare_wkly_benefit %>%
  filter(!is.na(chase_value),
         year == "2019",
         series == "median",
         !state_abb %in% vector_small_states) %>% #dropping small states, see #379
  mutate(has_monthly = if_else(has_monthly,
                               "Has monthly statistic",
                               "No monthly statistic")) %>%
  ggplot() +
  aes(x = dol_value,
      y = chase_value,
      label = state_abb) +
  geom_point(color = "#004577") +
  geom_text_repel(show.legend = FALSE) +
  scale_x_continuous(labels=function(x) paste0("$",x)) +
  scale_y_continuous(labels=function(x) paste0("$",x)) +
  geom_abline(slope = 1, linetype = "longdash") +
  fte_theme() +
  scale_colour_manual(values = uieip_palette[1:2]) +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 12)) +
  labs(x = "Mean Department of Labor weekly benefits, 2019",
       y = "",
       subtitle = "Median JPMCI weekly benefits, 2019") +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE))

ggsave(str_c(out_path_benchmarking, "weekly_benefits_median_2019_mthly.png"),
       width = 8, height = 4.5, units = "in")

