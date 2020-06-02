import torch
from nn import NeuralNet
import torch.nn as nn
import os
import sys
import random
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

root_dir = os.path.join("data")

classes_to_idx = {v: k for k, v in enumerate(open(os.path.join(root_dir, "classes.txt")).read().strip().split("\n"))}
idx_to_classes = {v: k for k, v in classes_to_idx.items()}

criterion = nn.BCELoss()
model = NeuralNet(0.01, criterion, 256, len(classes_to_idx))
model.load_state_dict(torch.load(sys.argv[1]))
if use_cuda:
    model.cuda()
model.eval()

id = random.choice(os.listdir(os.path.join(root_dir, "images"))).split(".")[0]
data = torch.load(os.path.join(root_dir, "images", id + ".pt"))
true_labels = torch.load(os.path.join(root_dir, "labels", id + ".pt"))

diseases = []
for x in range(len(true_labels)):
    if(true_labels[x] == 1.0):
        diseases.append(idx_to_classes[x])

print(id)
print(true_labels)
print(diseases)


with torch.no_grad():
    data_cuda, labels_cuda = data.to(device), true_labels.to(device)
    output_cuda = torch.round(model.forward(data_cuda.unsqueeze(0)))
    print(output_cuda.cpu().numpy())

# plt.imshow(data.permute(1,2,0))
# plt.show()
