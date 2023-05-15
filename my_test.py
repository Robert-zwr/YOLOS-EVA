import torch

model1 = torch.load('/home/zwr/YOLOS/checkpoints/eva/eva02_B_pt_in21k_p14to16.pt', map_location='cpu')
model2 = torch.load('/home/zwr/YOLOS/checkpoints/eva/EVA02_B_psz14to16.pt', map_location='cpu')

print(model1)
