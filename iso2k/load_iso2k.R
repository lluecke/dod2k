## --------------------------------------------------------------------------
##
## Script name: Iso2k guide code
##
## Purpose of script: Filter records from full Iso2k database
##
## Script author: G. Falster
##
## Date updated: 2020-06-10
##
## Email: gfalster@wustl.edu
##
## Citation: Konecky et al. (2020) The Iso2k Database: A global compilation of paleo-d18O and d2H records to aid understanding of Common Era climate
## Database is available for download from https://doi.org/10.25921/57j8-vs18 or http://lipdverse.org/iso2k/current_version/
## --------------------------------------------------------------------------
##
## Notes: 
##
## *current for database version 1.0.0
##   
## *written using R version 4.0.0 
##
## *for easier manipulation of LiPD files, we suggest installing the 'lipdR' package:
## introduction & installation instructions available at https://nickmckay.github.io/LiPD-utilities/r/index.html 
## examples in this script use only base R and a couple of commonly used packages.
##
## Code offered as-is. For questions or feedback on LiPD-utilities code, please contact Nick <nick@nau.edu>, or post an issue on github at https://github.com/nickmckay/lipd-utilities
## --------------------------------------------------------------------------

# =============================================================================
# set display options
# =============================================================================

options(scipen = 10, digits = 4) # run this if you don't like scientific notation in your outputs

# =============================================================================
# load required packages
# =============================================================================

library("magrittr")
library("tidyverse")

# =============================================================================
# load Iso2k database
# =============================================================================

load("iso2k1_0_1.RData") # change this to match the current R serialization.

#rm(D, TS) # Remove extraneous objects

# =============================================================================
# look at individual record
# =============================================================================

## first, a couple of ways to search by Iso2k unique identifier, or by site name.

## extract all Iso2k UIDs
TSids <- as.character(sapply(TS, "[[", "paleoData_iso2kUI"))

## filter chosen record from full record list
whichRecordName <- which(TSids == "MS12CCCH01b")
recordTS <- TS[whichRecordName]

## extract dataset names
siteNames <- as.character(sapply(TS, "[[", "geo_siteName"))

## filter datasets containing the desired site name from the full record list
whichSite <- which(grepl("bahamas", siteNames, ignore.case=TRUE))
selectedSiteTS <- TS[whichSite]

View(selectedSiteTS)

## view site names for the Bahamas records
as.character(sapply(selectedSiteTS, "[[", "geo_siteName")) %>%
  unique() %>%
  sort()

# =============================================================================
# initial filtering of the database, using Level 1 fields
# =============================================================================

## starting with the entire database, filter for records that
# have water isotope proxy data i.e.  d18O or d2H,
# have units in per mille, and
# are flagged as the primary timeseries

variableName <- as.character(sapply(TS, "[[", "paleoData_variableName"))
units <- as.character(sapply(TS, "[[", "paleoData_units"))
primaryTS <- as.character(sapply(TS, "[[", "paleoData_iso2kPrimaryTimeseries"))

## create filters for primary records with isotope data in per mille
isd18O <- which(variableName == "d18O" & primaryTS == "TRUE"  & units == "permil")
isd2H <- which(variableName == "d2H" & primaryTS == "TRUE"  & units == "permil")
isIso <- c(isd18O, isd2H)

allIsoTS <- TS[isIso] # apply filter to the full timeseries

length(allIsoTS) # See how many records are in this filtered subset of the database

# =============================================================================
# EXTRACT VARIABLES FOR export as csv:
# =============================================================================

# save metadata arrays (one value per record)
# =============================================================================

datasetId <- as.character(sapply(allIsoTS, "[[", "paleoData_TSid"))
write.table(datasetId, "iso2k1_0_1_csv/datasetId.csv",  
          col.names=FALSE, row.names = FALSE)
length(unique(datasetId))

archiveType <- as.character(sapply(allIsoTS, "[[", "archiveType"))
write.table(archiveType, "iso2k1_0_1_csv/archiveType.csv",  
          col.names=FALSE, row.names = FALSE)

climateInterpretation_variable <- as.character(sapply(allIsoTS, "[[", "interpretation1_variable"))
write.table(climateInterpretation_variable, "iso2k1_0_1_csv/climateInterpretation_variable.csv",  
          col.names=FALSE, row.names = FALSE)

dataSetName <- as.character(sapply(allIsoTS, "[[", "dataSetName"))
write.table(dataSetName, "iso2k1_0_1_csv/dataSetName.csv",  
          col.names=FALSE, row.names = FALSE)


geo_meanElev <- c(sapply(allIsoTS, "[[", "geo_elevation"))
write.table(geo_meanElev, "iso2k1_0_1_csv/geo_meanElev.csv",  
          col.names=FALSE, row.names = FALSE)

geo_meanLat <- c(sapply(allIsoTS, "[[", "geo_latitude"))
write.table(geo_meanLat, "iso2k1_0_1_csv/geo_meanLat.csv",  
          col.names=FALSE, row.names = FALSE)

geo_meanLon <- c(sapply(allIsoTS, "[[", "geo_longitude"))
write.table(geo_meanLon, "iso2k1_0_1_csv/geo_meanLon.csv",  
          col.names=FALSE, row.names = FALSE)

geo_siteName <- as.character(sapply(allIsoTS, "[[", "geo_siteName"))
write.table(geo_siteName, "iso2k1_0_1_csv/geo_siteName.csv",  
          col.names=FALSE, row.names = FALSE)

originalDataURL <- as.character(sapply(allIsoTS, "[[", "originalDataUrl"))
write.table(originalDataURL, "iso2k1_0_1_csv/originalDataURL.csv",  
          col.names=FALSE, row.names = FALSE)

paleoData_proxy <- as.character(sapply(allIsoTS, "[[", "paleoData_variableName"))
write.table(paleoData_proxy, "iso2k1_0_1_csv/paleoData_proxy.csv",  
          col.names=FALSE, row.names = FALSE)

paleoData_sensorSpecies <- as.character(sapply(allIsoTS, "[[", "paleoData_archiveSpecies"))
write.table(paleoData_sensorSpecies, "iso2k1_0_1_csv/paleoData_sensorSpecies.csv",  
          col.names=FALSE, row.names = FALSE)

paleoData_units <- as.character(sapply(allIsoTS, "[[", "paleoData_units"))
write.table(paleoData_units, "iso2k1_0_1_csv/paleoData_units.csv",  
          col.names=FALSE, row.names = FALSE)

yearUnits <- as.character(sapply(allIsoTS, "[[", "yearUnits"))
write.table(yearUnits, "iso2k1_0_1_csv/yearUnits.csv", 
          col.names=FALSE, row.names = FALSE)

paleoData_notes <- as.character(sapply(allIsoTS, "[[", "paleoData_notes"))
write.table(paleoData_notes, "iso2k1_0_1_csv/paleoData_notes.csv", 
          col.names=FALSE, row.names = FALSE)


# save numeric data (each record has an array of data with different length, )
# =============================================================================

# year
year <- as.list(sapply(allIsoTS, "[[", "year"))
dat = t(year[[1]])
write.table(dat,file="iso2k1_0_1_csv/year.csv",append=FALSE,
            sep=",",col.names=FALSE,row.names=FALSE)
for (i in 2:length(year)) {
  if (is.null(year[[i]])) {
    dat = c(-9999.99)
  }
  else {
    dat = t(year[[i]])
  }
  write.table(dat,file="iso2k1_0_1_csv/year.csv",append=TRUE,
              sep=",",col.names=FALSE,row.names=FALSE)
}

# paleoData_vales
paleoData_values <- as.list(sapply(allIsoTS, "[[", "paleoData_values"))
dat = t(paleoData_values[[1]])
write.table(dat,file="iso2k1_0_1_csv/paleoData_values.csv",append=FALSE,
            sep=",",col.names=FALSE,row.names=FALSE)

for (i in 2:length(paleoData_values)) {
  if (is.null(paleoData_values[[i]])) {
    dat = c(-9999.99)
  }
  else {
    dat = t(paleoData_values[[i]])
    
  }
  write.table(dat,file="iso2k1_0_1_csv/paleoData_values.csv",append=TRUE,
              sep=",",col.names=FALSE,row.names=FALSE)
}

print('Completed.')



