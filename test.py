import torch
import torch.nn as nn
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, classification_report
from tqdm.auto import tqdm
from nn import NeuralNet
from data_utils import *
import argparse
from torchsummary import summary

parser = argparse.ArgumentParser(description='Test set metrics on models')
parser.add_argument('--data_dir', help='directory where data is stored', required=True)
parser.add_argument('--model', help='path to model', required=True)
parser.add_argument('--dataset', help='choose either train, test, valid', required=True)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print("Using CUDA: {}".format(use_cuda))
torch.backends.cudnn.benchmark = True

root_dir = args.data_dir
model_name = args.model

# Parameters
params = {'batch_size': 1024,
          'shuffle': True}

train_x, train_y, test_x, test_y, valid_x, valid_y = parse_data(root_dir)
print("Loaded data...")

mean = torch.tensor(0.50745)
std = torch.tensor(0.25038)
print("Mean: {:.5f}\tstd: {:.5f}".format(mean.item(), std.item()))

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.item(), std=std.item())
])

data_x, data_y = None, None
if(args.dataset == "train"):
    data_x, data_y = train_x, train_y
elif(args.dataset == "test"):
    data_x, data_y = test_x, test_y
elif(args.dataset == "valid"):
    data_x, data_y = valid_x, valid_y

test_dataset = Dataset(root_dir, data_x, data_y, transforms=transform)
test_generator = torch.utils.data.DataLoader(test_dataset, **params)

print("Loaded dataloaders...")

criterion = torch.nn.CrossEntropyLoss()
model = NeuralNet(0.001, criterion, 64, 2)
state_dict = torch.load(model_name)
model.load_state_dict(state_dict)
for parameter in model.parameters():
    parameter.requires_grad = False
if(use_cuda):
    model.cuda()
model.eval()
summary(model, (1, 64, 64))

print("Loaded model...")

preds = []
labels = []

for local_batch, local_labels in tqdm(test_generator):
    labels.extend(local_labels.numpy().tolist())
    local_batch, local_labels = local_batch.to(device), local_labels.to(device)
    output = model.forward(local_batch)
    output = model.softmax(output)
    output = torch.max(output, 1)[1]
    preds.extend(output.cpu().numpy().tolist())

recall = recall_score(y_true=labels, y_pred=preds, average='weighted')
prec = precision_score(y_true=labels, y_pred=preds, average='weighted')
f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
acc = accuracy_score(labels, preds)
print("Accuracy: {}".format(acc))
print("Recall: {}\tPrecision: {}\tF1 Score: {}".format(recall, prec, f1))
print(classification_report(y_true=labels, y_pred=preds, target_names=["No Findings", "Pneumonia"]))
