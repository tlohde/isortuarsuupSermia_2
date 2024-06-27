#!/bin/sh

# for arctic dem shapefile catalogue
mkdir -p ../data/arcticDEM
wget --mirror --no-parent -r -P ../data/arcticDEM/ https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/indexes/ArcticDEM_Strip_Index_latest_gpqt.zip

# unzip any zip files created - in place
find ../ -name '*.zip' -type f -execdir unzip -n '{}' ';'