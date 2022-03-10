import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class preprocess_datasets():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
    def preprocess_dataset(self):
        '''
        Preprocess and 
        Attibutes
        ---------
        dataset_name: string, available names: 'german.data', 'synthetic.csv', 'BROWARD_CLEAN.csv', 'titanic.csv', 'adult.data'

        Returns
        -------
        DataFrame object including the preprocessed dataset
        '''

        if self.dataset_name == 'german.data':
            # retrieved from https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/
            
            df = pd.read_csv(self.dataset_name, na_values='?', header=None, sep=' ')
            cols = ['Status_of_existing_checking_account','Duration_in_month', 'Credit_history', 'Purpose', 'Credit_amount', 'Savings_accountbonds', 'Present_employment_since', 'Installment_rate_in_percentage_of_disposable_income', 'Personal_status_and_sex', 'Other_debtorsguarantors', 'Present_residence_since', 'Property', 'Age_in_years', 'Other_installment_plans', 'Housing', 'Number_of_existing_credits_at_this_bank', 'Job', 'Number_of_people_being_liable_to_provide_maintenance_for', 'Telephone', 'Foreign_worker', 'Creditworthiness']

            # sorting values for vizualization purposes
            df.columns = cols
            df = df.sort_values(by = 'Creditworthiness', axis=0, kind = 'stable', ignore_index=True)

            # normalize numeric variables
            numeric = [False if df[col].dtype == 'object' else True for col in df]
            nominal = [True if df[col].dtype == 'object' else False for col in df]
            num=df.loc[:,numeric].values[:,:-1]
            scaled=np.subtract(num,np.min(num,axis=0))/np.subtract(np.max(num,axis=0),np.min(num,axis=0))
            df[df.columns[numeric][:-1]] = pd.DataFrame(scaled, columns=df.columns[numeric][:-1])

            # recode 'Personal_status_and_sex' based on AIF360's preprocessing
            df['Personal_status_and_sex'] = np.where(df['Personal_status_and_sex'] == 'A92', 'female', 'male')

            return df
          
        if self.dataset_name == 'synthetic.csv':
            # refer Notebook in folder Python Scripts
            
            df = pd.read_csv(self.dataset_name, na_values='?')
            numeric = [False, True, False, False, True, False, False, True, False, False, True, False, True, False, False, True, False, True, False, False]
            df['Creditworthiness'] = np.where(df['Creditworthiness'] == 'good credit', 0, 1)
            Z = df.values

            for col,isnumeric in zip(df, numeric):
              if not isnumeric:
                df[col] = df[col].astype('object')

            #df.drop('Creditworthiness', axis=1, inplace=True)

            return df
        
        if self.dataset_name == 'BROWARD_CLEAN.csv':
            # retrieved from https://farid.berkeley.edu/downloads/publications/scienceadvances17/
            
            df = pd.read_csv(self.dataset_name, na_values='?')
            df = df.loc[:,df.columns[:-4]] # drop last three empty columns & compas_correct
            

            df.drop('id',axis=1,inplace=True)

            df['race'] = df['race'].astype('object')
            df['sex'] = df['sex'].astype('object')
            df['charge_id'] = df['charge_id'].astype('object')
            df['charge_degree (misd/fel)'] = df['charge_degree (misd/fel)'].astype('object')
            df['two_year_recid'] = df['two_year_recid'].astype('object')
            df['compas_guess'] = df['compas_guess'].astype('object')

            nom_cols = df.select_dtypes(include='object').columns
            num_cols = df.select_dtypes(exclude='object').columns

            df_num = df.loc[:,num_cols]
            scaled=np.subtract(df_num.values,np.min(df_num.values,axis=0))/np.subtract(np.max(df_num.values,axis=0),np.min(df_num.values,axis=0))
            df[num_cols] = pd.DataFrame(scaled, columns=df_num.columns)

            # predicted
            df.drop('two_year_recid',axis=1,inplace=True)
            numeric = [False if df[col].dtype == 'object' else True for col in df ]

            #df.drop('compas_guess', axis=1, inplace=True)
            df.rename(columns={"charge_degree (misd/fel)": "charge_degree"}, inplace=True)

            return df

        if self.dataset_name == 'titanic.csv':
            # retrieved from https://biostat.app.vumc.org/wiki/Main/DataSets (titanic3.csv)
            
            df = pd.read_csv(self.dataset_name, na_values=np.nan)

            # feature engineering
            df['title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            df['title'] = df['title'].replace('Mlle', 'Miss')
            df['title'] = df['title'].replace('Ms', 'Miss')
            df['title'] = df['title'].replace('Mme', 'Mrs')
            # drop PassengerId, Name, Ticket
            df.drop(['name', 'ticket', 'boat', 'home.dest', 'body', 'cabin'], axis=1, inplace=True)
            df['survived'] = df['survived'].astype(str)

            df = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare','embarked', 'title','survived']]

            df['age'] = np.where(df['age'] == '?',np.median(df['age'][df['age'] != '?'].astype('float64')),df['age']).astype('float64')
            df['fare'] = np.where(df['fare'] == '?',np.median(df['fare'][df['fare'] != '?'].astype('float64')),df['fare']).astype('float64')

            nom_cols = df.select_dtypes(include='object').columns
            num_cols = df.select_dtypes(exclude='object').columns

            df_num = df.loc[:,num_cols]
            scaled=np.subtract(df_num.values,np.min(df_num.values,axis=0))/np.subtract(np.max(df_num.values,axis=0),np.min(df_num.values,axis=0))
            df[num_cols] = pd.DataFrame(scaled, columns=df_num.columns)

            numeric = [True if i == 'float64' else False for i in df.dtypes.values]

            # df.drop('survived',axis=1,inplace=True)

            return df

        if self.dataset_name == 'adult.data':
            # https://archive.ics.uci.edu/ml/datasets/adult
            
            df = pd.read_csv(self.dataset_name, na_values='?', header=None)
            names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
                    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
                    'hours-per-week', 'native-country', 'income']

            df.columns = names

            nom_cols = df.select_dtypes('object').columns
            num_cols = df.select_dtypes(exclude='object').columns

            df_num = df.loc[:,list(df[num_cols].columns)]
            scaled=np.subtract(df_num.values,np.min(df_num.values,axis=0))/np.subtract(np.max(df_num.values,axis=0),np.min(df_num.values,axis=0))
            df[num_cols] = pd.DataFrame(scaled, columns=df_num.columns)

            df['income'] = np.where(df['income']==" <=50K",0,1)
            df_nom = df.loc[:,list(df[nom_cols[:-1]].columns)]

            return df
