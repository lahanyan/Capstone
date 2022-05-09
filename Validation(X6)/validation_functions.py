def data_prep(data):
    import pandas as pd
    import numpy as np
    
    filepath = "../Data/important_50_X.csv"

    imp_var = pd.read_csv(filepath)

    NAs_cols = ['NumberOfBankruptciesFiled',
            'NumberOfBankruptciesDischarged',
            'NumberOfBankruptciesDismissed',
            'NumberOfBankruptciesDisposed',
            'mortgageinquiriespast3months',
            'mortgageinquiriespast6months',
            'HighestbalanceofopenVAloan_1MO',
            'CumulativebalancesofopenVAloan_1MO',
            '30daylatepast3months_1MO',
            '60daylatethepast3months_1MO',
            '90dayormorelatepast3months_1MO',
            '30daylatepast6months_1MO',
            '60daylatepast6months_1MO',
            '90ormoredaylatepast6months_1MO',
            '30daylatepast12months_1MO',
            '60daylatepast12months_1MO',
            '90dayormorelatepast12months_1MO',
            'dayspastduecurrently_1MO',
            'FHAloans_1MO',
            'openVAloans_1MO',
            'openFinanceMortgageloans_1MO',
            'MortgageOpenTradeLines',
            'MortgageSumPayment',
            'MortgageSumBalance',
            'Numberof60DPDwithinthelast12months',
            'Numberof60DPDwithinthelast18months',
            'Numberof90DPDwithinthelast12months',
            'Numberof90DPDwithinthelast18months',
            'WeeksSinceLastTarget',
            'DaysSinceLastTarget',
            'AgeOfOldestOpenAndCurrentRevolvingTrade',
            'NumOfOpenAndCurrentFinanceTrades',
            'HighestRevolvingCreditAmount',
            'KeycodedAggBalCredLimitRatioForOpenRevolvingTrades',
            'MinBalToCreditOpenAuto',
            'MaxBalToCreditOpenAuto',
            'FICOTier',
            'TargetedInLast30',
            'TargetedInLast60',
            'TargetedInLast90',
            'TargetedInLast180',
            'TimesTargetedLast30',
            'TimesTargetedLast60',
            'TimesTargetedLast90',
            'TimesTargetedLast180',
            'TimesTargeted',
            'AnnualPercentageRate',
            'Payment',
            'MonthsSinceOpen']
    
    for col in NAs_cols:
        if (col in data.columns) and (col in imp_var.cols) :
            data[col] = data[col].fillna(0)
    
    
    mean_cols = [   'Age',
                'FICO5Score',
                'FICO8Score',
                'FICO8AutoScore',
                'FICO9Score',
                'FICO9AutoScore',
                'Monthlypaymentamountofhighestmortgagetrade_1MO',
                'Cumulativemonthlypaymentsforallopenmortgagetrades_1MO',
                'BalanceofopenFHAloan_1MO',
                'CumulativebalancesofopenFHAloan_1MO',
                'AgeofmostrecentFHAmortgagetrade_1MO',
                'AgeofmostrecentVAmortgagetrade_1MO',
                'BalanceofopenFinanceMortgage_1MO',
                'CumulativebalancesofopenFinanceloan_1MO',
                'AgeofmostrecentFinancemortgagetrade_1MO',
                'MortgageSumHighCredit',
                'PErsonalLoanSumPayment',
                'PErsonalLoanSumHighCredit',
                '3MonthFICO5Delta',
                '6MonthFICO5Delta',
                '12MonthFICO5Delta',
                '18MonthFICO5Delta',
                '3MonthFICOAutoDelta',
                '6MonthFICOAutoDelta',
                '12MonthFICOAutoDelta',
                '18MonthFICOAutoDelta',
                'MonthsSinceMostRecentMortgageOpened',
                'MortgageLTV']
    
    for col in mean_cols:
        if (col in data.columns) and (col in imp_var.cols):
            mean_value=data[col].mean()
            data[col].fillna(value=mean_value, inplace=True)
        
        
    obj_cols_to_transform = ['MailType',
                         'OpenTradeLinesAuto',
                         'MortgageWorstDelinqEverReptdStatusCodeValue',
                         'Typecodeformostrecentlyopenedmortgagetrade_1MO']
    
    for col in obj_cols_to_transform:
        col_values = data[col].unique()
        for val in col_values:
            data[col + "_" + str(val)] = np.where(data[col] == val, 1, 0)
        
    data = data.drop(obj_cols_to_transform, axis=1)
    
    data_validation = data[imp_var.cols]
    
    
    data_validation[['ZIPCode', 'Individual_ID']] = data[['ZIPCode', 'Individual_ID']]
    
    return data_validation



def mf_sample(data_validation):
    import sys
    import sklearn.neighbors._base
    sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
    #pip install misspy for installation
    from missingpy import MissForest
    import pandas as pd
    import numpy as np
    
    data_validation_sample = data_validation.sample(n=1000, random_state=42)
    
    imputer = MissForest(random_state=42)
    data_imputed = imputer.fit_transform(data_validation_sample)
    
    scaled_data_for_mf = pd.DataFrame(data_imputed, index=data_validation_sample.index, columns=data_validation_sample.columns)
    data_validation_sample[scaled_data_for_mf.columns] = scaled_data_for_mf
    
    return data_validation_sample



def dict_final(data_validation_sample):
    import pandas as pd

    dict_final = {'ZIPCode': data_validation_sample['ZIPCode'], 'Individual_ID': data_validation_sample['Individual_ID']}
    data_final = pd.DataFrame(dict_final)

    return data_final

def drop_zip_id(data_validation_sample):
    import pandas as pd
    return data_validation_sample.drop(['ZIPCode', 'Individual_ID'], axis=1)

def get_users(data_validation_sample, data_final):
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    filepath = "../Data/TrainDataMf.csv"

    train_data_mf = pd.read_csv(filepath) 
    
    x_train_mf = train_data_mf.drop(['Lead Flag', 'ZIPCode'], axis=1)
    y_train_mf = train_data_mf['Lead Flag']
    
    model_mf = lgb.LGBMClassifier(learning_rate=0.1,max_depth=3,random_state=42)
    model_mf.fit(x_train_mf,y_train_mf, verbose=20)
    
    predict = model_mf.predict_proba(data_validation_sample)[:,1]
    
    data_final['Lead Flag'] = np.where(predict > 0.8, 1, 0)
    data_final['prediction'] = predict
    data_final = data_final[data_final['Lead Flag'] == 1]
    
    return data_final