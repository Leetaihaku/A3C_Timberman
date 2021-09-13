import os, subprocess
import pickle

import Main_Header as Main
import Client_Worker as CW

import torch
from collections import deque

batch = deque()
for _ in range(8):
    a = []
    a.append(torch.tensor([123456., 61., 0., 0., 0], device='cuda'))
    a.append(1)
    a.append(torch.tensor([123443., 61., 0., 0., 0.], device='cuda'))
    a.append(torch.ones(1, device='cuda'))
    batch.append(None)
    batch[_] = a

#CW.transmit_batch(batch)
data = pickle.dumps(batch)
print(data)
print(str(data))
