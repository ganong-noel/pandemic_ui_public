# =========================================== #
# driver.R
# =========================================== #

# setting wd
rm(list = ls())
setwd("~/repo/pandemic_ui_public/")

# Load devtools for installing specific versions
library(devtools)

# Function to install a specific package version and handle errors
install_specific_version <- function(pkg, version) {
  tryCatch({
    install_version(pkg, version = version)
    message(paste("Successfully installed", pkg, "version", version))
  }, error = function(e) {
    message(paste("Error installing", pkg, "version", version, ":", e$message))
  })
}


# Install specific versions of packages
packages <- list(
  "RColorBrewer" = "1.1-3",
  "yaml" = "2.3.7",
  "testthat" = "3.1.9",
  "scales" = "1.2.1",
  "readxl" = "1.4.2",
  "ggrepel" = "0.9.3",
  "geojsonio" = "0.11.1",
  "broom" = "1.0.0",
  "lubridate" = "1.9.2",
  "tidyverse" = "2.0.0"
)

# Loop through the packages and install specified versions
lapply(names(packages), function(pkg) {
  install_specific_version(pkg, packages[[pkg]])
})

# load packages
library(tidyverse)
library(lubridate)
library(broom)
library(geojsonio)
library(ggrepel)
library(readxl)
library(rprojroot)
library(scales)
library(testthat)
library(yaml)
library(RColorBrewer)
versions <- sessionInfo()$otherPkgs %>% map_chr(c("Version"))
print(versions)


test_that(
  "record package versions",
  {
    expect_equal(versions["RColorBrewer"] %>% unname(), "1.1-3")
    #expect_equal(versions["yaml"] %>% unname(), "2.3.7")
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
library(rgeos)

test_that(
  "record package versions",
  {
    expect_equal(versions["rgeos"] %>% unname(), "0.6-3")
  }
)

# =========================================== #

