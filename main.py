from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from pydantic import BaseModel
import joblib
import numpy as np
from clearml import Task, Model
import joblib

from preprocess import InputData , process_input

app = FastAPI()

class MyResponse(BaseModel):
    label: int
    score: float



""" data = InputData(
    SERVICE_UNIT_QUANTITY=10,
    Age=84,
    TOTAL_CHARGES=1638.06,
    BILL_TYPE_CODE=131,
    HCPCS_CODE='G8979',
    GENDER='female',
    RACE='white',
    STATE='Utah',
    COUNTY='COUNTY_X',
    CODE='CODE_X',
    PRESENT_ON_ADMIT=0,
    DUAL_STATUS=0,
    ENCOUNTER_TYPE='Other',
    PAYERS='Medicare',
    PROCEDURE_CODE='0W3P8ZZ',
    PLACE_OF_SERVICE_CODE=23,
    REVENUE_CENTER_CODE=636,
    DISCHARGE_DISPOSITION_CODE=1,
    MEDICARE_STATUS=10,
    ADMIT_SOURCE_CODE=2,
    ADMIT_TYPE_CODE=1,
    MS_DRG=470,
    PROC_DESC='PD_1'
) """


#@app.on_event("startup")
def load_model():

    global model
    global scaler
    global label_encoder

    model_path = Model(model_id="6e5051c914dc438cbb7b67fa43c96912").get_local_copy()
    scaler_path = Model(model_id='703b88f1617046499ebf12880ac6126c').get_local_copy()
    label_encoder_path = Model(model_id='1d9c4a4430444db7b0e22432dda8a07d').get_local_copy()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)

@app.post("/predict")
def predict(input_data: InputData):
    load_model()

    in_data = process_input(input_data,scaler)
    print(in_data.shape)

    prediction_prob = model.predict_proba(in_data)

    index = prediction_prob[0].argmax()
    print(f'index :{index}')

    index_np = np.array(index)

    prediction  = label_encoder.inverse_transform(index_np.reshape(1,1))
    #print(f'prediction :{prediction}')
    print(prediction)

    resp = MyResponse( label= prediction, score=prediction_prob[0,index])
    json_compatible_item_data = jsonable_encoder(resp)

    return JSONResponse(content=json_compatible_item_data)






    """ print(f'Predicted Prob;{prediction}')
    print(f'Predicted index :{prediction[0].argmax()}')
    prediction  = label_encoder.inverse_transform(prediction[0].argmax())
    print(f"prediction: {prediction}") """

    #return {"prediction":prediction}
#predict(data)
