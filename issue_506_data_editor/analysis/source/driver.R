# =========================================== #
# driver.R
# =========================================== #

# setting wd
rm(list = ls())
setwd("~/repo/pandemic_ui_public/")

#PACKAGES TO INSTALL EARLIER RELEASES OF PACKAGES
library(renv)

# Initialize renv
renv::init(project = "~/repo/pandemic_ui_public/issue_506_data_editor", bare = TRUE)
renv::activate(project = "~/repo/pandemic_ui_public/issue_506_data_editor")

renv::install("devtools")
renv::install("testthat")

# set library paths to use renv library
.libPaths("~/repo/pandemic_ui_public/issue_506_data_editor/renv/library")

test_that("the correct library is being used", {
  library_paths <- .libPaths()
  expect_true(
    any(grepl("~/repo/pandemic_ui_public/issue_506_data_editor/renv/library", library_paths)),
    info = paste("Library paths are:", paste(library_paths, collapse = ", "))
  )
})

install.packages(testthat)
install.packages(devtools)

# libraries to install earlier versions of packages
# note: you will be prompted after installaiotion to see if you want to 
#       update your pacakges. Select '3' to say 'no.'
install_version("RColorBrewer", version = "1.1-3")
install_version("yaml", version = "2.3.7")
install_version("testthat", version = "3.1.9")
install_version("scales", version = "1.2.1")
install_version("readxl", version = "1.4.2")
install_version("ggrepel", version = "0.9.3")
install_version("geojsonio", version = "0.11.1")
install_version("broom", version = "1.0.0")
install_version("lubridate", version = "1.9.2")
install_version("tidyverse", version = "2.0.0")

# Snapshot the project's dependencies
renv::snapshot()

# load packages
library(tidyverse)
library(lubridate)
library(broom)
library(geojsonio)
library(ggrepel)
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
    expect_equal(versions["readxl"] %>% unname(), "1.4.2")
    expect_equal(versions["ggrepel"] %>% unname(), "0.9.3")
    expect_equal(versions["geojsonio"] %>% unname(), "0.11.1")
    expect_equal(versions["broom"] %>% unname(), "1.0.0")
    expect_equal(versions["lubridate"] %>% unname(), "1.9.2")
    expect_equal(versions["tidyverse"] %>% unname(), "2.0.0")
  }
)

# setup paths
config <- yaml::yaml.load_file("~/repo/pandemic_ui_public/analysis/config.yml")
source("~/repo/pandemic_ui_public/prelim.R")

input_path <- "~/repo/pandemic_ui_public/analysis/input"
chase_path <- paste0(input_path, "/disclose")

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

job_find_file <- str_c(
  chase_path,
  "/2023-06-23_disclosure_packet/",
  "jobfind/",
  "2023-06-18_ui_jobfind_for_export.xls"
)

# benchmarks and plots
code_path <- source_path
source(file.path(code_path, "diagnostic_benchmarking_plots.R"))



# ============================================================================ #
#                -- RUN ONLY IF NEED TO RECREATE HEXAMAP --
# ============================================================================ #

# Please use R 3.6.3 to run this script
# =========================================== #
# -- STEP 1 -- 
# macOS: 
#    You can download R 3.6.3 from the CRAN archive and install it.
# Windows: 
#    Download R 3.6.3 from the CRAN archive.
# Linux: 
#   Use the following commands to install R 3.6.3:
#      sudo apt-get update
#      sudo apt-get install r-base=3.6.3-1bionic
#      sudo apt-mark hold r-base
#
# -- STEP 2 --
# Configure RStudio to use R 3.6.3:
# 1. Go to Tools > Global Options.
# 2. Under General, you will find the R version. Click Change and select R 3.6.3.
# =========================================== #

# Check R version
if (getRversion() != "3.6.3") {
  stop("This script requires R 3.6.3. Please switch to R 3.6.3 and try again.")
}

# these packages must be installed, but are not available in the most recent 
# version of R as of 2023-07-15
install.packages("colortools")
install.packages("rgdal")

# need to install GEOS (Geometry Engine - Open Source) library
# macOS
# > brew install geos
# linux 
# > sudo apt-get install libgeos-dev
# windows
# > download and install the binary from https://libgeos.org/
remotes::install_version("rgeos", version = "0.6-3")

test_that(
  "record package versions",
  {
    expect_equal(versions["rgeos"] %>% unname(), "0.6-3")
  }
)

# =========================================== #

