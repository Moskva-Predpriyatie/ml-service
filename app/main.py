from fastapi import FastAPI
from .GBEstimator import GBEstimator
from .FCLHak import FCLHak
import numpy as np

app = FastAPI()

gradient = GBEstimator("model/GBEstimator")
neural = FCLHak("model/FCLHakmodel")

@app.get("/getGradientBoosting")
async def getGradientBoosting(orgType: int, personnel: int, equipment: int, district: int, landSquare: int, facilitySquare: int, isLandRental: int, isFacilityRental: int, rentalSquare: int):
    data_vector = np.array([ orgType, personnel, equipment, district, landSquare, facilitySquare, isLandRental, isFacilityRental, rentalSquare ])
    predict = gradient.predict(data_vector)
    return { 
        "equipmentCost": str(predict[0]),
        "facilityCost": str(predict[1]),
        "landCost": str(predict[2]),
        "rentalCost": str(predict[3]),
        "salaryCost": str(predict[4]),
        "insuranceCost": str(predict[5]),
        "from": str(predict[6]),
        "to": str(predict[7])
    }

@app.get("/getNeuralNetwork")
async def getGradientBoosting(orgType: int, personnel: int, equipment: int, district: int, landSquare: int, facilitySquare: int, isLandRental: int, isFacilityRental: int, rentalSquare: int):
    data_vector = np.array([ orgType, personnel, equipment, district, landSquare, facilitySquare, isLandRental, isFacilityRental, rentalSquare ])
    predict = neural.predict(data_vector)
    return { 
        "equipmentCost": str(predict[0]),
        "facilityCost": str(predict[1]),
        "landCost": str(predict[2]),
        "rentalCost": str(predict[3]),
        "salaryCost": str(predict[4]),
        "insuranceCost": str(predict[5]),
        "from": str(predict[6]),
        "to": str(predict[7])
    }
