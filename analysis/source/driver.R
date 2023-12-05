# driver.R

rm(list = ls())
setwd("~/repo/pandemic_ui_public_internal")

# load packages
library(tidyverse)
library(glue)
library(lubridate)
library(gt)
library(broom)
library(geojsonio)
library(ggrepel)
library(ipumsr)
library(magick)
library(mapproj)
library(readxl)
library(rgeos)
library(rprojroot)
library(scales)
library(testthat)
library(yaml)
library(RColorBrewer)
versions <- sessionInfo()$otherPkgs %>% map_chr(c("Version"))

test_that(
  "record package versions",
  {
    expect_equal(versions["RColorBrewer"] %>% unname(), "1.1-3")
    expect_equal(versions["yaml"] %>% unname(), "2.3.7")
    expect_equal(versions["testthat"] %>% unname(), "3.1.9")
    expect_equal(versions["scales"] %>% unname(), "1.2.1")
    expect_equal(versions["rprojroot"] %>% unname(), "2.0.3")
    expect_equal(versions["rgeos"] %>% unname(), "0.6-3")
    expect_equal(versions["sp"] %>% unname(), "2.0-0")
    expect_equal(versions["readxl"] %>% unname(), "1.4.2")
    expect_equal(versions["mapproj"] %>% unname(), "1.2.11")
    expect_equal(versions["maps"] %>% unname(), "3.4.1")
    expect_equal(versions["magick"] %>% unname(), "2.7.4")
    expect_equal(versions["ipumsr"] %>% unname(), "0.5.2")
    expect_equal(versions["ggrepel"] %>% unname(), "0.9.3")
    expect_equal(versions["geojsonio"] %>% unname(), "0.11.1")
    expect_equal(versions["broom"] %>% unname(), "1.0.0")
    expect_equal(versions["gt"] %>% unname(), "0.9.0")
    expect_equal(versions["glue"] %>% unname(), "1.6.2")
    expect_equal(versions["lubridate"] %>% unname(), "1.9.2")
    expect_equal(versions["forcats"] %>% unname(), "1.0.0")
    expect_equal(versions["stringr"] %>% unname(), "1.5.0")
    expect_equal(versions["dplyr"] %>% unname(), "1.1.2")
    expect_equal(versions["purrr"] %>% unname(), "1.0.1")
    expect_equal(versions["readr"] %>% unname(), "2.1.4")
    expect_equal(versions["tidyr"] %>% unname(), "1.3.0")
    expect_equal(versions["tibble"] %>% unname(), "3.2.1")
    expect_equal(versions["ggplot2"] %>% unname(), "3.4.2")
    expect_equal(versions["tidyverse"] %>% unname(), "2.0.0")
  }
)

# setup paths
make_path <- rprojroot::is_git_root$make_fix_file()
config <- yaml::yaml.load_file(make_path("analysis/config.yml"))
source(paste0(config$template_repo, "prelim.R"))
out_path <- make_path("analysis/release")
chase_path <- make_path("analysis/input/disclose")

# color palettes
uieip_palette <- c(
  "#004577",
  "#8a89c4",
  "#bbd976",
  "#ffae5f",
  "#a2dadb"
)

# current disclosures
# ui_benchmarking_new.R uses the disclosure from 2022-06-29
job_find_file <- str_c(
  chase_path,
  "/2023-06-23_disclosure_packet/",
  "jobfind/",
  "2023-06-18_ui_jobfind_for_export.xls"
)

# benchmarks and plots
code_path <- "analysis/source"
source(file.path(code_path, "diagnostic_benchmarking_plots.R"))
