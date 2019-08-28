"""
Remove sensitive data and make generic
"""
import json
import pandas as pd
import os
import numpy as np

cwd = os.getcwd()
print(cwd)

__location__ = ""
#os.path.dirname(os.path.realpath(__file__))


"""Load data"""
df_data_name = __location__  + "input_data/data_ROI_brand_generic.json"
df_all_name = __location__ + "input_data/no_ROI_brand_generic.json"

with open(df_data_name, "r") as of_d:
    data = json.load(of_d)
    df_data = pd.io.json.json_normalize(data)

with open(df_all_name, "r") as of_a:
    data = json.load(of_a)
    df_all = pd.io.json.json_normalize(data)


"""Remove Sensitive Info"""
df_data.drop(columns=['sku', 'countRedeemers', 'unused', 'countRedeemers'], inplace=True)


"""Change brands"""
df_combined = df_data.append(df_all)

#Create new brands names
brands = df_combined["brand"].tolist()
unique = list(set(brands))
new_brands = ["BRAND_"+str(i) for i in np.arange(len(unique))]
brands_dict = dict(zip(unique, new_brands))


#Change brands using dictionary
df_data = df_data.replace({"brand": brands_dict})
df_all = df_all.replace({"brand": brands_dict})


"""Change programs"""
df_combined = df_data.append(df_all)

#Create new programs names
programs = df_combined["program"].tolist()
uniq = list(set(programs))
new_programs = ["PROGRAM_"+str(i) for i in np.arange(len(uniq))]
brands_prog = dict(zip(uniq, new_programs))

#Change brands using dictionary
df_data = df_data.replace({"program": brands_prog})
df_all = df_all.replace({"program": brands_prog})


"""
Save new data to .JSON:
"""
def outputter(df_output, name):
    #File writing code
    out = df_output.to_json(orient='records')
    out_filename = os.path.join(__location__, name)
    with open(out_filename, 'w') as f:
        f.write(out)
    f.close()

outputter(df_data, df_data_name)
outputter(df_all, df_all_name)

print("Files outputted")

