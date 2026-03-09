import pandas as pd

df1 = pd.read_csv('prep_times.csv')
df2 = pd.read_csv('pred_times-test.csv')

df = df1.merge(df2, how='left', on='pdbid')

df.to_csv('CASF2016_ConBAP_preds_and_times.csv', index=False)
