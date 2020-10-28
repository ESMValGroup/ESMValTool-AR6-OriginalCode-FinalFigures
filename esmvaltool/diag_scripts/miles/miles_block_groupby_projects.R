# #############################################################################
# miles_block_groupby_projects.r
# Authors:       P. Davini (ISAC-CNR, Italy) (author of MiLES)
# 	         J. von Hardenberg (ISAC-CNR, Italy) (ESMValTool adaptation)
#                E. Arnone (ISAC-CNR, Italy) (ESMValTool v2.0 adaptation)
#                R. Kazeroni (DLR) (project grouping for TM90)
# #############################################################################
# Description
# MiLES is a tool for estimating properties of mid-latitude climate.
# It works on daily 500hPa geopotential height data and it produces
# climatological figures for the chosen time period. Data are interpolated
# on a common 2.5x2.5 grid.
# Model data are compared against a reference field such as the
# ECMWF ERA-Interim reanalysis.
#
# Modification history
#   20201028-kazeroni_remi: modified to average over projects (CMIP5, OBS...)
#   20181203-vonhardenberg_jost: Completed conversion, rlint compliant
#   
# ############################################################################

library(tools)
library(yaml)

provenance_record <- function(infile) {
  xprov <- list(
    ancestors = infile,
    authors = list(
      "vonhardenberg_jost", "davini_paolo",
      "arnone_enrico"
    ),
    references = list(
      "davini18", "davini12jclim",
      "tibaldi90tel"
    ),
    projects = list("c3s-magic"),
    caption = "MiLES blocking statistics",
    statistics = list("other"),
    realms = list("atmos"),
    themes = list("phys"),
    domains = list("nh")
  )
  return(xprov)
}

diag_scripts_dir <- Sys.getenv("diag_scripts")

source(paste0(diag_scripts_dir, "/miles/basis_functions.R"))
source(paste0(diag_scripts_dir, "/miles/block_figures.R"))
source(paste0(diag_scripts_dir, "/miles/block_fast.R"))
source(paste0(diag_scripts_dir, "/miles/miles_parameters.R"))
source(paste0(diag_scripts_dir, "/shared/external.R")) # nolint

# read settings and metadata files
args <- commandArgs(trailingOnly = TRUE)
settings <- yaml::read_yaml(args[1])
metadata <- yaml::read_yaml(settings$input_files)
for (myname in names(settings)) {
  temp <- get(myname, settings)
  assign(myname, temp)
}

field_type0 <- "T2Ds"

# get first variable and list associated to pr variable
var0 <- "zg"
list0 <- metadata

# get name of climofile for first variable and list
# associated to first climofile
climofiles <- names(list0)
climolist0 <- get(climofiles[1], list0)

diag_base <- climolist0$diagnostic
print(paste(diag_base, ": starting routine"))

# create working dirs if they do not exist
work_dir <- settings$work_dir
regridding_dir <- settings$run_dir
plot_dir <- settings$plot_dir
dir.create(work_dir, recursive = T, showWarnings = F)
dir.create(regridding_dir,
  recursive = T,
  showWarnings = F
)
dir.create(plot_dir, recursive = T, showWarnings = F)

# setup provenance file and list
provenance_file <-
  paste0(regridding_dir, "/", "diagnostic_provenance.yml")
provenance <- list()

# extract metadata
models_dataset <- unname(sapply(list0, "[[", "dataset"))
models_ensemble <- unname(sapply(list0, "[[", "ensemble"))
models_exp <- unname(sapply(list0, "[[", "exp"))
models_projects <- unname(sapply(list0, "[[", "project"))
models_start_year <- unname(sapply(list0, "[[", "start_year"))
models_end_year <- unname(sapply(list0, "[[", "end_year"))
models_experiment <- unname(sapply(list0, "[[", "exp"))
models_ensemble <- unname(sapply(list0, "[[", "ensemble"))

##
## Run it all
##

for (model_idx in c(1:(length(models_dataset)))) {
  exp <- models_exp[model_idx]
  dataset <- models_dataset[model_idx]
  ensemble <- models_ensemble[model_idx]
  year1 <- models_start_year[model_idx]
  year2 <- models_end_year[model_idx]
  infile <- climofiles[model_idx]
  project <- models_projects[model_idx]
  for (seas in seasons) {
    filenames <- miles_block_fast(
      year1 = year1,
      year2 = year2,
      expid = exp,
      ens = ensemble,
      dataset = dataset,
      season = seas,
      z500filename = infile,
      FILESDIR = work_dir,
      doforce = TRUE
    )
    # Set provenance for output files
    xprov <- provenance_record(list(infile))
    for (fname in filenames) {
      provenance[[fname]] <- xprov
    }
  }
}

##
## Make the plots
##
if (write_plots) {

  for (seas in seasons) {
    filenames <- miles_block_figures_groupby(
      models_exp = models_exp,
      models_ens = models_ensemble,
      models_project = models_project,
      models_year1 = models_start_year,
      models_year2 = models_end_year,
      seasons = seas,
      FIG_DIR = plot_dir,
      FILESDIR = work_dir,
      REFDIR = work_dir
    )
    # Set provenance for output files (same as diagnostic files)
    xprov <- provenance_record(list(infile)) # not correct !!!
    for (fname in filenames$figs) {
      provenance[[fname]] <- xprov
    }
  }
}

# Write provenance to file
write_yaml(provenance, provenance_file)
