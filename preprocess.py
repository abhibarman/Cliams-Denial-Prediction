#dependencies

#recieve input from request

#prepprocess
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

encoded_data = pd.read_csv('X_train.csv')

class InputData(BaseModel):
    
      TOTAL_CHARGES:float
      SERVICE_UNIT_QUANTITY:int
      Age:int
      BILL_TYPE_CODE:str
      HCPCS_CODE:str
      GENDER:str
      RACE:str
      STATE:str
      COUNTY:str
      CODE:str
      PRESENT_ON_ADMIT:str
      DUAL_STATUS:int
      ENCOUNTER_TYPE:str
      PAYERS:str
      PROCEDURE_CODE:str
      PROC_DESC:str
      PLACE_OF_SERVICE_CODE:int
      REVENUE_CENTER_CODE:int
      DISCHARGE_DISPOSITION_CODE:int
      MEDICARE_STATUS:int
      ADMIT_SOURCE_CODE:int
      ADMIT_TYPE_CODE:int
      MS_DRG:int

def process_input(inputData:InputData,scaler:MinMaxScaler):

      new_data = dict()
      new_data['TOTAL_CHARGES'] = inputData.TOTAL_CHARGES
      new_data['SERVICE_UNIT_QUANTITY'] = inputData.SERVICE_UNIT_QUANTITY
      new_data['Age'] = inputData.Age
      new_data['BILL_TYPE_CODE'] = inputData.BILL_TYPE_CODE
      new_data['HCPCS_CODE'] = inputData.HCPCS_CODE
      new_data['GENDER'] = inputData.GENDER
      new_data['RACE'] = inputData.RACE
      new_data['STATE'] = inputData.STATE
      new_data['COUNTY'] = inputData.COUNTY
      new_data['CODE'] = inputData.CODE
      new_data['PRESENT_ON_ADMIT'] = inputData.PRESENT_ON_ADMIT
      new_data['DUAL_STATUS'] = inputData.DUAL_STATUS
      new_data['ENCOUNTER_TYPE'] = inputData.ENCOUNTER_TYPE
      new_data['PAYERS'] = inputData.PAYERS
      new_data['PROCEDURE_CODE'] = inputData.PROCEDURE_CODE      
      new_data['PLACE_OF_SERVICE_CODE'] = inputData.PLACE_OF_SERVICE_CODE
      new_data['REVENUE_CENTER_CODE'] = inputData.REVENUE_CENTER_CODE
      new_data['DISCHARGE_DISPOSITION_CODE'] = inputData.DISCHARGE_DISPOSITION_CODE
      new_data['MEDICARE_STATUS'] = inputData.MEDICARE_STATUS
      new_data['ADMIT_SOURCE_CODE'] = inputData.ADMIT_SOURCE_CODE
      new_data['ADMIT_TYPE_CODE'] = inputData.ADMIT_TYPE_CODE
      new_data['MS_DRG'] = inputData.MS_DRG
      new_data['PROC_DESC'] = inputData.PROC_DESC


      new_data = pd.DataFrame(new_data, index=[0])
      
      dct = dict(inputData)
      num_cols = ['Age', 'SERVICE_UNIT_QUANTITY', 'TOTAL_CHARGES'] 
      scaled = scaler.transform(new_data[num_cols])

      new_data['Age'] =  scaled[:,0]
      new_data['TOTAL_CHARGES'] =  scaled[:,1]
      new_data['SERVICE_UNIT_QUANTITY'] = scaled[:,2]      

      catcols = [k for k,v in dct.items() if k not in num_cols]
      new_encoded = pd.get_dummies(new_data, columns=catcols)

      for col in encoded_data.columns:
            if col not in new_encoded.columns:
                  new_encoded[col] = 0

      print(set(new_encoded.columns.tolist()).difference(encoded_data.columns.tolist()))
      drop_firsts = set(new_encoded.columns.tolist()).difference(encoded_data.columns.tolist())
      new_encoded.drop(columns=drop_firsts, inplace=True)
      print(set(encoded_data.columns.tolist()).difference(new_encoded.columns.tolist()))

      return new_encoded[encoded_data.columns] #new_encoded.reindex(columns=encoded_data.columns)