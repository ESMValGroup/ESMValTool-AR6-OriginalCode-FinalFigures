; #############################################################################
; EXAMPLE SCRIPT FOR NCL DIAGNOSTICS
; #############################################################################
; Author: Mattia Righi (DLR, Germany), Klaus-Dirk Gottschaldt (DLR, Germany)
; #############################################################################
;
; Description
;    Simple script to demonstrate the implementation of a NCL diagnostic.
;    It shows how the select the metadata for specific variable(s) and
;    dataset(s) and how to read the corresponding fields.
;    The input is averaged over time to create a global contour plot.
;    Output to NetCDF files is optional (and controlled via diag_script_info)
;    Time averaging and plotting is done in separate routines.
;    All information necessary for plotting is added as attributes to the
;    variable containing the data. Attribute names "diag_script", "var"
;    and those starting with "res_" are reserved for that purpose.
;    The data variable (incl. metadata) can be passed to the plot script
;    directly (most efficient), or via a netCDF file (might be useful for
;    non-NCL routines).
;    A workstation ("wks" = file to plot to) will by default be generated
;    in the plotting routine. However, the default will be overridden, if
;    a "graphic" object is passed to the plotting routine. The latter
;    option is intended to be used in special cases, e.g. with
;    res_gsnDraw = False and res_gsnFrame = False.
;
; Required diag_script_info attributes
;    projection: map projection to use
;    colormap: color table to use
;    ncdf: full path and name for *.nc output
;
; Optional diag_script_info attributes
;    MyDiag_title: string for annotation
;
; Caveats
;
; Modification history
;    20181107-righi_mattia: revised and extended to include new metadata
;                           handling
;    20151027-lauer_axel: moved call to 'write_references' to the beginning
;                         of the code
;    20150325-lauer_axel: modified reference tags used for acknowledgements
;                         (projects, observations, etc)
;    20140312-righi_mattia: adapted to new time_operations structure
;    20121130-gottschaldt_klaus-dirk: written
;
; #############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"  ; load metadata

load "$diag_scripts/shared/statistics.ncl"  ; load statistics functions
load "$diag_scripts/shared/scaling.ncl"

load "$diag_scripts/shared/plot/style.ncl"  ; load plot style functions
load "$diag_scripts/shared/plot/contour_maps.ncl"  ; load plot function

begin

  ; DIAG_SCRIPT is a global variables defined in interface.ncl
  enter_msg(DIAG_SCRIPT, "")

  ; Shortcuts to current variable and field
  ; For multiple variables, use different indices of variable_info
  var0 = variable_info[0]@short_name
  info0 = select_metadata_by_name(input_file_info, var0)

  ; The number of variables is the size of the variable_info list
  n_variables = ListCount(variable_info)

end


begin

  ; The metadata for this diagnostic are defined in the input_file_info list
  ; in interface.ncl for all variables and datasets: use the convenience
  ; function to extract the metadata for a specific variable or dataset

  ; Retrieve metadata items by attribute (short_name and field)
  atts = True
  atts@short_name = var0
  info = select_metadata_by_atts(input_file_info, atts)
  delete(atts)

  ; Simple version using only the variable's short name (works for most cases)
  info_simple = select_metadata_by_name(input_file_info, var0)

  ; The number of datasets for var0 is the size of the returned list
  n_datasets = ListCount(info)

  ; Read data for the first dataset in the list
  ; Loop over all n_datasets info items to read all datasets in a loop
  dataset = read_data(info[0])
  printVarSummary(dataset)

  ; Retrieve metadata item for the reference dataset
  atts = True
  atts@short_name = var0
  atts@dataset = variable_info[0]@reference_dataset
  info_ref = select_metadata_by_atts(input_file_info, atts)
  delete(atts)

  ; Read data for the reference dataset
  reference = read_data(info_ref[0])
  printVarSummary(reference)

  ; Calculate time-average of the reference dataset
  ; A function in esmaltool/diag_scripts/shared/statistics.ncl is used
  timeavg = time_operations(reference, -1, -1, "average", "annualclim", True)

  if (var0.eq."pr") then
    timeavg = convert_units(timeavg, "mm/day")
  end if

  ; Set annotations
  annots = project_style(info_ref, diag_script_info, "annots")


  ; Other Metadata: diag_script, var
  ; Add to timeavg, as attributes without prefix
  if(isatt(timeavg, "diag_script")) then  ; Add to existing entries
    temp = timeavg@diag_script
    delete(timeavg@diag_script)
    timeavg@diag_script = array_append_record(temp, (/diag_script/), 0)
    delete(temp)
  else  ; Add as new attribute
    timeavg@diag_script = (/DIAG_SCRIPT/)
  end if
  timeavg@var = var0  ; Overwrite existing entry
  timeavg@var_units = timeavg@units
  if(isatt(variable_info, "long_name")) then
    timeavg@var_long_name = variable_info[0]@long_name
  end if


  ; Plot resources
  timeavg@res_gsnMaximize  = True  ; use full page for the plot
  timeavg@res_cnFillOn  = True  ; color plot desired
  timeavg@res_cnLineLabelsOn  = False  ; contour lines
  timeavg@res_cnLinesOn  = False
  timeavg@res_tiMainOn  = True
  timeavg@res_gsnLeftStringFontHeightF = 0.015
  timeavg@res_gsnRightStringFontHeightF = 0.015
  timeavg@res_cnLevelSelectionMode  = "ExplicitLevels"
  timeavg@res_mpOutlineOn  = True
  timeavg@res_mpFillOn  = False
  timeavg@res_lbLabelBarOn  = True
  timeavg@res_gsnRightString  = ""
  timeavg@res_gsnLeftString   = ""
  timeavg@res_mpFillDrawOrder  = "PostDraw"  ; draw map fill last
  timeavg@res_cnMissingValFillColor  = "Gray"
  timeavg@res_tmYLLabelsOn  = False
  timeavg@res_tmYLOn  = False
  timeavg@res_tmYRLabelsOn  = False
  timeavg@res_tmYROn  = False
  timeavg@res_tmXBLabelsOn  = False
  timeavg@res_tmXBOn  = False
  timeavg@res_tmXTLabelsOn  = False
  timeavg@res_tmXTOn  = False
  timeavg@res_cnInfoLabelOn  = False  ; turn off cn info label
  timeavg@res_mpProjection  = diag_script_info@projection

  timeavg@res_gsnDraw  = False  ; Do not draw yet
  timeavg@res_gsnFrame  = False  ; Don't advance frame.
  timeavg@res_mpPerimOn  = False

  timeavg@res_mpGeophysicalLineColor = "gray42"
  timeavg@res_mpPerimOn    = False
  timeavg@res_mpGridLineColor   = "transparent"
  timeavg@res_mpGridAndLimbOn   = True
  timeavg@res_mpOutlineOn = True

  timeavg@res_lbTitleString  = "(" + timeavg@units + ")"
  timeavg@res_lbTitlePosition  = "Bottom"

  timeavg@res_lbLabelFontHeightF  = 0.014
  timeavg@res_lbTopMarginF  = 0.1
  timeavg@res_lbTitleFontHeightF  = 0.014

  timeavg@res_lbBoxEndCapStyle = "TriangleBothEnds"
  timeavg@res_lbBoxLineColor = "gray70"

  timeavg@res_tiMainFontHeightF  = 0.016

  timeavg@res_gsnRightStringParallelPosF = 0.8

  if(isatt(diag_script_info, "caption")) then
    timeavg@res_tiMainString = diag_script_info@caption
  else
    timeavg@res_tiMainString = "Climatology"
  end if

  timeavg@res_cnLevels = ispan(1, 10, 1)

  pal = read_colormap_file("$diag_scripts/shared/plot/rgb/" \
                           + "ipcc-ar6_precipitation_seq.rgb")
  timeavg@res_cnFillPalette = pal


  ; Output to netCDF
  timeavg@ncdf = diag_script_info@ncdf
  ncdf_outfile = ncdf_write(timeavg, diag_script_info@ncdf)


  ; Create the plot

  if (isvar("wks")) then
    delete(wks)
  end if

  wks = get_wks("dummy_for_wks", DIAG_SCRIPT, "clim_" + var0 \
                  + "_annualclim_" + input_file_info[0]@project)

  map = contour_map(wks, timeavg, var0)

  res = True

  line1 = gsn_add_polyline(wks, map, (/0.,180./), (/30.,30./), res)
  line2 = gsn_add_polyline(wks, map, (/-180.,0./), (/30.,30./), res)
  line3 = gsn_add_polyline(wks, map, (/0.,180./), (/60.,60./), res)
  line4 = gsn_add_polyline(wks, map, (/-180.,0./), (/60.,60./), res)
  line5 = gsn_add_polyline(wks, map, (/0.,180./), (/-30.,-30./), res)
  line6 = gsn_add_polyline(wks, map, (/-180.,0./), (/-30.,-30./), res)
  line7 = gsn_add_polyline(wks, map, (/0.,180./), (/-60.,-60./), res)
  line8 = gsn_add_polyline(wks, map, (/-180.,0./), (/-60.,-60./), res)
 
  draw(map)
  frame(wks)

  ; Call provenance logger
  log_provenance(ncdf_outfile, \
                 map@outfile, \
                 "Mean of variable: " + var0, \
                 "mean", \
                 "global", \
                 "geo", \
                 (/"righi_mattia", "gottschaldt_klaus-dirk"/), \
                 (/"acknow_author"/), \
                 metadata_att_as_array(info0, "filename"))

end
