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
## Plotting parameters
##
if (write_plots) {
  color_field <- c("dodgerblue", "darkred", "black")
  color_diff <- NULL
  lev_field <- c(0, 30)
  lev_diff <- NULL
  lev_hist <- NULL
  legend_unit <- "Blocked Days (%)"
  legend_distance <- 3
  title_name <- "TM90 Instantaneous Blocking"
  fp <- list(
    color_field = color_field,
    color_diff = color_diff,
    lev_field = lev_field,
    lev_diff = lev_diff,
    lev_hist = lev_hist,
    legend_unit = legend_unit,
    legend_distance = legend_distance,
    title_name = title_name
  )
  alpha <- 50 # transparency coefficient
  # panels option
  par(
    cex.main = 2,
    cex.axis = 1.5,
    cex.lab = 1.5,
    mar = c(5, 5, 4, 3),
    oma = c(0, 0, 0, 0)
  )
  lwdline <- 4
  tm90cols <- fp$color_field

##
## Compute mean and std for datasets grouped by project
##
  field <- "TM90"
  project_list <- c("CMIP6", "CMIP5")
  i_project <- 1
  for (season in seasons) {
    field_exp_all <- c() # store field for all datasets
    for (model_idx in c(1:(length(models_dataset)))) {
      expid <- models_exp[model_idx]
      dataset <- models_dataset[model_idx]
      ens <- models_ensemble[model_idx]
      year1 <- models_start_year[model_idx]
      year2 <- models_end_year[model_idx]
      project <- models_project[model_idx]
      # use file.builder function
      nomefile <- file_builder(
      FILESDIR,
      "Block",
      "BlockClim",
      dataset,
      expid,
      ens,
      year1,
      year2,
      season
      )
      field_exp <- ncdf_opener(nomefile, namevar = field, rotate = "no")
      assign(paste(field, "_exp", sep = ""), field_exp)
      field_exp_all <- c(field_exp_all, list(field_exp))
    }

    filenames <- c()
    # create figure names with ad-hoc function
    figname <- fig_builder( ## Adjust this!!
      FIGDIR,
      "Block",
      field,
      dataset,
      expid,
      ens,
      year1,
      year2,
      season,
      output_file_type
    )
    filenames <- c(filenames, figname)
    open_plot_device(figname, output_file_type, special = TRUE)
    text_legend <- c() # text of the legend
    for (project in project_list){
      field_exp_mean <- field_exp_all[models_project == project]
      field_mean <- list(apply(X=as.data.frame(field_exp_mean), MARGIN = 1, FUN = mean)) # mean over datasets of the project
      field_std <- list(apply(X=as.data.frame(field_exp_mean), MARGIN = 1, FUN = sd)) # std
      n_datasets <- length(field_exp_mean) # number of datasets for the project
      text_legend <- c(text_legend, paste(project, ":", n_datasets, "datasets"))
      mycol <- rgb(aperm(col2rgb(tm90cols[i_project])), max = 255, alpha = alpha) # adjust color for the shaded area (std)

      # rotation to simplify the view (90 deg to the west)
      n <- (-length(ics) / 4)
      ics2 <- c(tail(ics, n), head(ics, -n) + 360)
      field_mean2 <- c(tail(field_mean, n), head(field_mean, -n))
      field_std2 <- c(tail(field_std, n), head(field_std, -n))
      field_mean_up2 <- unlist(field_mean2) + unlist(field_std2)
      field_mean_do2 <- unlist(field_mean2) - unlist(field_std2)

      if (i_project == 1){
        plot(
          ics2,
          unlist(field_mean2),
          type = "l",
          lwd = lwdline,
          ylim = fp$lev_field,
          main = fp$title_name,
          xlab = "Longitude",
          ylab = fp$legend_unit,
          col = tm90cols[1]
        )
        grid()
      }
      else {
        points(
          ics2,
          unlist(field_mean2),
          type = "l",
          lwd = lwdline,
          col = tm90cols[i_project]
        )
      }
      polygon(c(ics2, rev(ics2)), c(field_mean_do2, rev(field_mean_up2)), col = mycol, border = NA) # shaded area [mean-std; mean+std]
      i_project <- i_project + 1
    }
    legend(
     100,
     30,
      legend = text_legend,
      lwd = lwdline,
      lty = 1,
      col = tm90cols[1:length(project_list)],
      bg = "white",
      cex = 1.
    )
    dev.off()
    ## Set provenance for output files (same as diagnostic files)
    #xprov <- provenance_record(list(
    #  climofiles[model_idx],
    #  climofiles[ref_idx]
    #))
    #for (fname in filenames$figs) {
    #  provenance[[fname]] <- xprov
    #}
  }
}

# Write provenance to file
write_yaml(provenance, provenance_file)
