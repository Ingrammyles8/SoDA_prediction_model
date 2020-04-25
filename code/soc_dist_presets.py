#soc_dist_presets
"""
Created on Tue Apr 21 18:35:05 2020

@author: mai2125
"""
import pathlib as pl
import pandas as pd
import numpy as np

# path to folders
src = pl.Path.cwd().parent
us_demo = src/"US_Demographics"
cvd_time = src/"csse_covid_19_time_series"
mobi = src/"Mobility_Data"
outputs = src/"outputs"

# master datasheet
m_dataset = pd.read_csv(us_demo/"master_soc_dist_dataset_trim.csv") 

# state abbreviation to name dictionary
abbr_dict = {"AK": "Alaska", "AL":"Alabama", "AR":"Arkansas", "AZ":"Arizona", "CA":"California",
             "CO":"Colorado", "CT":"Connecticut", "DE":"Delaware", "DC":"District of Columbia", "FL": "Florida",
             "GA":"Georgia", "HI":"Hawaii", "IA":"Iowa", "ID":"Idaho", "IL":"Illinois",
             "IN":"Indiana", "KS":"Kansas", "KY":"Kentucky", "LA":"Louisiana", "MA": "Massachusetts",
             "MD":"Maryland", "ME": "Maine", "MI": "Michigan", "MN": "Minnesota", "MO":"Missouri",
             "MN":"Minnesota", "MT": "Montana", "NE":"Nebraska", "NV":"Nevada", "NH":"New Hampshire",
             "NJ":"New Jersey", "NC":"North Carolina", "NM":"New Mexico", "NY":"New York",
             "ND":"North Dakota", "OH":"Ohio", "OK":"Oklahoma", "OR":"Oregon", "PA":"Pennsylvania",
             "RI":"Rhode Island", "SC":"South Carolina", "SD":"South Dakota", "TN":"Tennessee",
             "TX":"Texas", "UT":"Utah", "VT":"Vermont", "VA":"Virginia", "WA":"Washington",
             "WV":"West Virginia", "WI":"Wisconsin", "WY":"Wyoming"}

# US Demographic dataframes
census_table = pd.ExcelFile(us_demo/"US_Demographics.xlsx")
census_data = census_table.parse("Sheet1")
census_data = census_data.set_index("Fact").T.reset_index().rename(columns={"index":"states"})
for i in range(len(census_data.columns)):
    census_data.iloc[:, i] = pd.to_numeric(census_data.iloc[:, i], errors='ignore')
master_table = pd.ExcelFile(us_demo/"data_states.xlsx")
states_data = master_table.parse("Master")
for i in range(len(states_data.columns)):
    states_data.iloc[:, i] = pd.to_numeric(states_data.iloc[:, i], errors='ignore')

# covid timeseries spreadsheet. Goes till April 19th
cvd_cases_csv = cvd_time/"time_series_covid19_confirmed_US.csv"
mobi_csv = mobi/"Global_Mobility_Report.csv"

# parses Google's mobility data into only 
global_mobi_df = pd.read_csv(mobi_csv)

# extracts only the state level mobility data from the spreadsheet
us_mobi_df = global_mobi_df[global_mobi_df.loc[:, "country_region_code"] == "US"]
states_df = us_mobi_df[us_mobi_df.loc[:, "sub_region_2"].isnull()].dropna(subset=["sub_region_1"])
states_df.loc[:, "date"] = pd.to_datetime(states_df["date"])
states_df = states_df.rename(columns={"sub_region_1":"states"}).drop("sub_region_2", axis=1)

# date of national stay at home order
nsah_order = np.datetime64('2020-03-16')
states_df.loc[:, "stay_at_home_days"] = (states_df["date"]-nsah_order).dt.days

# aggregates all the mobility trends for public places into one number
soda = ((states_df.iloc[:,4] + states_df.iloc[:, 5] + states_df.iloc[:, 6]
                + states_df.iloc[:, 7] + states_df.iloc[:, 8])/5)

states_df.loc[:, "SoDA Score"] = soda

# looks at only the days after the national stay at home order was issued
states_q_df = states_df[states_df["stay_at_home_days"] >= 0]

# aggregates the mean mobility trends over those days into one number
mobi_df = states_q_df.groupby("states").agg({"SoDA Score":"mean"}).reset_index()
mobi_df.loc[:, "SoDA Score"] = mobi_df.loc[:, "SoDA Score"].abs()
#mobi_df.to_csv("states_mobility_mean.csv")

# covid timeseries dataframe from challenge github
cvd_cases = pd.read_csv(cvd_cases_csv)

# creates a dictionary for every date in the dataframe columns
cvd_dates = cvd_cases.columns[11:]
first_dict = {dates:"sum" for dates in cvd_dates}
first_case = cvd_cases.groupby("Province_State").agg(first_dict).reset_index()

# cumulative covid cases per state
cvd_state = cvd_cases.groupby("Province_State").agg({"4/19/20":"sum"}).reset_index().rename(columns={"4/19/20":"cumulative_cases", 
                             "Province_State":"states"})





