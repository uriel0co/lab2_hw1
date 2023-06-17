import os
import pandas as pd


def clip_hr_and_remove_weird_patients(df, test=True):
    if df['SepsisLabel'].max() == 0:
        if df['ICULOS'].max() > 72  and not test:
            return False

        if df['HR'].max() > 114:
            df['HR'] = 114

        if df['HR'].max() > 114:
            df['HR'] = 114

def operate(df, dict, columns, oper):
    if oper == 'max':
        for col in columns:
            dict[col] = df[col].max()
    if oper == 'min':
        for col in columns:
            dict[col] = df[col].min()
    if oper == 'avg':
        for col in columns:
            dict[col] = df[col].mean()
    return dict

def create_df(data_dir, name, test):
    create_df = True
    for filename in os.listdir(data_dir):
        dict_f_df = {}
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath, sep='|')
        df = df.head(len(df[df['SepsisLabel']==0])+1)
        
        df_filtered = clip_hr_and_remove_weird_patients(df, test)
        if df_filtered == False:
            continue
        
        dict_f_df = operate(df, dict_f_df, ['ICULOS', 'SepsisLabel','HR','SBP','Age','Gender','Temp'], 'max')
        dict_f_df = operate(df, dict_f_df, ['O2Sat','HospAdmTime'], 'min')
        dict_f_df = operate(df, dict_f_df, ['Resp', 'MAP'], 'avg')
        
        dict_f_df['pid'] = int(filename.split("_")[1].split(".")[0])
        nulls = df.isna().mean()
        dict_f_df['nulls'] = nulls.mean()

        cols = list(dict_f_df.keys())
        if create_df:
            df_final = pd.DataFrame(columns=cols)
            create_df = False

        df_final = pd.concat([df_final, pd.DataFrame([dict_f_df])], ignore_index=True)
    df_final = df_final.fillna(df_final.mean())
    df_final.to_csv(name+'.csv', index=False)
    return df_final

def train_resample(data, r):
    
    is_sick_patient = data['SepsisLabel'] == 1
    is_healthy_patient = data['SepsisLabel'] == 0

    sick_data = data[is_sick_patient].groupby('pid').filter(lambda x: x['SepsisLabel'].any())
    healthy_data = data[is_healthy_patient].groupby('pid').filter(lambda x: x['SepsisLabel'].all() == 0)
    num_rows_fit = int(r * healthy_data.shape[0])
    num_rows_needed = num_rows_fit - sick_data.shape[0]

    selected_rows = sick_data.sample(n=num_rows_needed, replace=True)
    balanced_sick_data = pd.concat([sick_data, selected_rows]).reset_index(drop=True)
    balanced_data = pd.concat([balanced_sick_data, healthy_data]).sample(frac=1).reset_index(drop=True)

    return balanced_data
