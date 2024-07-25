# =========================================== #
# driver.R
# =========================================== #

# setting wd
rm(list = ls())
setwd("~/repo/pandemic_ui_public/")

# load packages
library(tidyverse)
library(lubridate)
library(broom)
library(geojsonio)
library(ggrepel)
library(readxl)
library(scales)
library(testthat)
library(yaml)
library(RColorBrewer)

#check versions
versions <- sessionInfo()$otherPkgs %>% map_chr(c("Version"))
print(versions)

#test that versions are correct
test_that(
  "record package versions",
  {
    expect_equal(versions["RColorBrewer"] %>% unname(), "1.1-3")
    expect_equal(versions["yaml"] %>% unname(), "2.3.9")
    expect_equal(versions["testthat"] %>% unname(), "3.1.9")
    expect_equal(versions["scales"] %>% unname(), "1.3.0")
    expect_equal(versions["readxl"] %>% unname(), "1.4.2")
    expect_equal(versions["ggrepel"] %>% unname(), "0.9.5")
    expect_equal(versions["geojsonio"] %>% unname(), "0.11.3")
    expect_equal(versions["broom"] %>% unname(), "1.0.6")
    expect_equal(versions["lubridate"] %>% unname(), "1.9.3")
    expect_equal(versions["tidyverse"] %>% unname(), "2.0.0")
  }
)

# setup paths
config <- yaml::yaml.load_file("~/repo/pandemic_ui_public/analysis/config.yml")
source("~/repo/pandemic_ui_public/prelim.R")

input_path <- "~/repo/pandemic_ui_public/analysis/input"
chase_path <- str_c(input_path, "/disclose")

source_path <- "~/repo/pandemic_ui_public/issue_506_data_editor/analysis/source"

release_path <- "~/repo/pandemic_ui_public/issue_506_data_editor/analysis/release"
out_path <- release_path

# color palettes
uieip_palette <- c(
  "#004577",
  "#8a89c4",
  "#bbd976",
  "#ffae5f",
  "#a2dadb"
)

job_find_file <- paste0(
  chase_path,
  "/2023-06-23_disclosure_packet/",
  "jobfind/",
  "2023-06-18_ui_jobfind_for_export.xls"
)

# benchmarks and plots
code_path <- source_path
source(file.path(code_path, "diagnostic_benchmarking_plots.R"))

