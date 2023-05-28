from joblib import load
import warnings
import numpy as np


def warn(*args, **kwargs):
    pass

warnings.warn = warn

class GBEstimator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = load(model_path)

    def predict(self, client_input):
        client_input = client_input.astype(float)
        index_input = np.array([24, 10142.0, 100, 11, 2192246.751841947, 361001.015125, 1, 1, 1000])
        index_out = np.array([8.36286438e+05, 4.33201218e+07, 6.33671116e+06, 9.49121850e+02, 9.77688800e+05, 2.93306640e+05, 3.90177795e+07, 5.27887604e+07])
        client_input = client_input / index_input
        answers = self.model.predict(client_input.reshape(1, -1))
        answers = answers[0] * index_out

        return abs(answers)
