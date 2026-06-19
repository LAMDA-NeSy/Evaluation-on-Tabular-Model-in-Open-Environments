#!/bin/bash
# bash -e download_data.sh 

# 2018 1-Year
mkdir -p acs/2018/1-Year
wget -P acs/2018/1-Year https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pca.zip
unzip -o acs/2018/1-Year/csv_pca.zip -d acs/2018/1-Year

wget -P acs/2018/1-Year https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_ppr.zip
unzip -o acs/2018/1-Year/csv_ppr.zip -d acs/2018/1-Year

wget -P acs/2018/1-Year https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pla.zip
unzip -o acs/2018/1-Year/csv_pla.zip -d acs/2018/1-Year


wget -P acs/2018/1-Year https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pms.zip
unzip -o acs/2018/1-Year/csv_pms.zip -d acs/2018/1-Year

wget -P acs/2018/1-Year https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_phi.zip
unzip -o acs/2018/1-Year/csv_phi.zip -d acs/2018/1-Year

wget -P acs/2018/1-Year https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pne.zip
unzip -o acs/2018/1-Year/csv_pne.zip -d acs/2018/1-Year

# 2010 1-Year
mkdir -p acs/2010/1-Year
wget -P acs/2010/1-Year https://www2.census.gov/programs-surveys/acs/data/pums/2010/1-Year/csv_pca.zip
unzip -o acs/2010/1-Year/csv_pca.zip -d acs/2010/1-Year

# 2017 1-Year
mkdir -p acs/2017/1-Year
wget -P acs/2017/1-Year https://www2.census.gov/programs-surveys/acs/data/pums/2017/1-Year/csv_pca.zip
unzip -o acs/2017/1-Year/csv_pca.zip -d acs/2017/1-Year
