import numpy as np
import torch
import torch.nn as nn
from joblib import load
from torch import cuda

class CustomFCL(nn.Module):
    def __init__(self):
        super(CustomFCL, self).__init__()

        self.len = 8
        self.fc1 = nn.Linear(9, 50)
        self.fc2 = nn.Linear(50, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 200)
        self.fc5 = nn.Linear(200, self.len)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class FCLHak():
    def __init__(self, model_path):
        self.model = CustomFCL()
        self.model_path = model_path
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, client_input):
        device = 'cuda' if cuda.is_available() else 'cpu'
        with torch.no_grad():
            self.model.eval()
            client_input = client_input.astype(float)
            index_input = np.array([2.40000000e+01, 1.01420000e+04, 1.00000000e+02, 1.10000000e+01,
                                    2.19224675e+06, 3.61001015e+05, 1.00000000e+00, 1.00000000e+00,
                                    1.00000000e+03])
            index_out = np.array([8.36286438e+05, 4.33201218e+07, 6.33671116e+06, 9.49121850e+02,
                                  9.77688800e+05, 2.93306640e+05, 3.90177795e+07, 5.27887604e+07])
            client_input = client_input / index_input
            final = torch.Tensor(client_input).to(device)
            output = self.model(final)
            output = abs(output.detach().cpu().numpy())
            answer = index_out * output
        return answer
