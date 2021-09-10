import os, subprocess
import Main_Header as Main
import Client_Worker as CW

import torch
from collections import deque

batch = deque()
for _ in range(10):
    a = []
    a.append(torch.tensor([123456., 61., 0., 0., 0], device='cuda'))
    a.append(torch.tensor([123456., 61., 0., 0., 0], device='cuda'))
    a.append(torch.tensor([123456., 61., 0., 0., 0], device='cuda'))
    a.append(torch.tensor([123456., 61., 0., 0., 0], device='cuda'))
    a.append(torch.tensor([123456., 61., 0., 0., 0], device='cuda'))
    a.append(torch.tensor([123456., 61., 0., 0., 0], device='cuda'))
    a.append(torch.tensor([123456., 61., 0., 0., 0], device='cuda'))
    a.append(torch.tensor([123456., 61., 0., 0., 0], device='cuda'))
    a.append(torch.tensor([123456., 61., 0., 0., 0], device='cuda'))
    a.append(torch.tensor([123456., 61., 0., 0., 0], device='cuda'))
    batch.append(None)
    batch[_] = a

CW.transmit_batch(batch)
