import pandas as pd
import os
import glob
#load prep csv in dataframe
dataframe = pd.read_csv("C:\\Users\\rahli\\PycharmProjects\\pythonProject\\ML_Seminar_PISA\\WB_prep\\WDICSV.csv", sep=";")

#drop the years not needed
r = range(1960, 1998)
td = list(map(str,r))
dropped_early_years = dataframe.drop(columns=td)
clean_years_dataframe = dropped_early_years.drop(columns=['2019', '2020'])

#drop column "Country Name"
dropped_cname_df = clean_years_dataframe.drop(columns=['Country_Name'])

#drop the countrys not needed
#load math_pisa into data frame
pisa_frame = pd.read_csv("C:\\Users\\rahli\\PycharmProjects\\pythonProject\\ML_Seminar_PISA\\WB_prep\\20-11-05_PISA_math.csv", sep=";")
country_list_pisa = pisa_frame['LOCATION'].tolist()
clean_country_set_pisa = (set(country_list_pisa))
#list of countries in new_dataframe
country_list_df = dropped_cname_df['Country_Code'].tolist()
clean_country_set_df = set(country_list_df)
#compare the two lists and delete the needed countries
country_noneed = [i for i in clean_country_set_df if i not in clean_country_set_pisa]
#dropping the country codes not needed and their rows
for c in country_noneed:
    dropped_cname_df = dropped_cname_df[dropped_cname_df.Country_Code != c]

#Merge Column Indicator_Name and Indicator_Code
dropped_cname_df["Indicator_Code-Indicator_Name"] = dropped_cname_df["Indicator_Code"] + '-' + dropped_cname_df["Indicator_Name"]

#drop Columns Indicator_Name and Indicator_Code
dropped_cname_df = dropped_cname_df.drop(columns=['Indicator_Name', 'Indicator_Code'])

#split Dataframe by Country_Code
#first delete all duplicates in Country_Code and delete 'OAVG' and 'TWN'
country_list_pisa_wod = []
for d in country_list_pisa:
    if d not in country_list_pisa_wod:
        country_list_pisa_wod.append(d)

country_list_pisa_wod.remove('OAVG')
country_list_pisa_wod.remove('TWN')

df = {}
grouped = dropped_cname_df.groupby(dropped_cname_df.Country_Code)
for cc in country_list_pisa_wod:
    df[cc] = grouped.get_group(cc)

# Add Row for Country_Code
for m in country_list_pisa_wod:
    new_row = pd.Series(data= {'Country_Code':m, '1998':m, '1999':m, '2000':m, '2001':m, '2002':m, '2003':m, '2004':m, '2005':m, '2006':m,'2007':m, '2008':m, '2009':m, '2010':m, '2011':m, '2012':m, '2013':m, '2014':m, '2015':m,'2016':m, '2017':m, '2018':m, 'Indicator_Code-Indicator_Name':m }, name='Country_Code')
    df[m] = df[m].append(new_row, ignore_index=False)
    # get the 'Indicator_Code-Indicator_Name' - Column as Index
    df[m] = df[m].set_index('Indicator_Code-Indicator_Name')
    df[m] = df[m].rename(index= {m:'Country_Code'})
    #delete the Country_Code Columns
    df[m] = df[m].drop(columns=['Country_Code'])
    #Add Row Year
    new_r = pd.Series(data={'1998':'1998', '1999':'1999', '2000':'2000', '2001':'2001', '2002':'2002', '2003':'2003', '2004':'2004', '2005':'2005', '2006':'2006','2007':'2007', '2008':'2008', '2009':'2009', '2010':'2010', '2011':'2011', '2012':'2012', '2013':'2013', '2014':'2014', '2015':'2015','2016':'2016', '2017':'2017', '2018':'2018'}, name='Year')
    df[m] = df[m].append(new_r, ignore_index=False)
    #rename the Columns to y-cc
    df[m] = df[m].rename (columns= {'1998':f'1998-{m}', '1999':f'1999-{m}', '2000':f'2000-{m}', '2001':f'2001-{m}', '2002':f'2002-{m}', '2003':f'2003-{m}', '2004':f'2004-{m}', '2005':f'2005-{m}', '2006':f'2006-{m}','2007':f'2007-{m}', '2008':f'2008-{m}', '2009':f'2009-{m}', '2010':f'2010-{m}', '2011':f'2011-{m}', '2012':f'2012-{m}', '2013':f'2013-{m}', '2014':f'2014-{m}', '2015':f'2015-{m}','2016':f'2016-{m}', '2017':f'2017-{m}', '2018':f'2018-{m}'} )

final_df = pd.DataFrame()
#Merge all the single Country dfs
for m in country_list_pisa_wod:
        final_df = pd.concat([final_df, df[m]], axis=1 )

#Transpose Dataframe
final_df = final_df.T

#Move Yaer and CC to front of Dataframe
cols = list(final_df)
cols.insert(0, cols.pop(cols.index('Country_Code')))

final_df = final_df.ix[:, cols]
cols = list(final_df)
cols.insert(0, cols.pop(cols.index('Year')))

final_df = final_df.ix[:, cols]
#safe Dataframe as csv
new_dataframe_csv = final_df.to_csv('C:\\Users\\rahli\\PycharmProjects\\pythonProject\\ML_Seminar_PISA\\WB_prep\\WB_final.csv', index=True)

