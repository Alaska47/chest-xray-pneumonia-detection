import torch
import torch.nn as nn
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, classification_report
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from nn import NeuralNet
from data_utils import *
from utils import *
from torchsummary import summary
from datetime import date

today = date.today()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print("Using CUDA: {}".format(use_cuda))
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 1024,
          'shuffle': True,
          'num_workers':6}
max_epochs = 100

root_dir = os.path.join("data")
model_dir = os.path.join("models")

print("Normalizing dataset...")
# normalize dataset here
'''
# Run 1 time, and then cache results
loader = torch.utils.data.DataLoader(training_dataset, batch_size=4096, shuffle=False)
mean = 0.0
for images, _ in tqdm(loader):
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
mean = mean / len(loader.dataset)
var = 0.0
for images, _ in tqdm(loader):
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    var += ((images - mean.unsqueeze(1))**2).sum([0,2])
std = torch.sqrt(var / (len(loader.dataset)*256*256))
'''
mean = torch.tensor(0.50745)
std = torch.tensor(0.25038)
print("Mean: {:.5f}\tstd: {:.5f}".format(mean.item(), std.item()))

transforms = {
    "train": transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(64),
        # transforms.RandomChoice([transforms.RandomHorizontalFlip(), transforms.RandomAffine(20), transforms.ColorJitter(brightness=(1.2, 1.5))]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.item(), std=std.item())
    ]),
    "val": transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.item(), std=std.item())
        ])
}

train_x, train_y, test_x, test_y, valid_x, valid_y = parse_data(root_dir)
print("Loaded data...")

train_dataset = Dataset(root_dir, train_x, train_y, transforms=transforms["train"])
valid_dataset = Dataset(root_dir, valid_x, valid_y, transforms=transforms["val"])
training_generator = torch.utils.data.DataLoader(train_dataset, **params)
validation_generator = torch.utils.data.DataLoader(valid_dataset, **params)
print("Created datasets...")

criterion = torch.nn.CrossEntropyLoss()
model = NeuralNet(0.001, criterion, 64, 2)
if(use_cuda):
    model.cuda()
summary(model, (1, 64, 64))

print("Starting training...")
writer = SummaryWriter()

global_train_step = 0
global_val_step = 0
# Loop over epochs
for epoch in range(max_epochs):
    tqdm.write("Epoch: {}".format(epoch))

    progress_bar = tqdm(total=len(train_dataset), leave=True, position=0)

    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        loss, yhat = model.step(local_batch, local_labels)
        yhat = model.softmax(yhat)
        yhat = torch.max(yhat, 1)[1].cpu()
        acc = pred_acc(local_labels.cpu(), yhat)
        writer.add_scalar('Train/Accuracy', acc, global_train_step)
        writer.add_scalar('Train/Loss', loss, global_train_step)
        global_train_step += params['batch_size']
        progress_bar.update(params['batch_size'])
        progress_bar.set_postfix(loss=loss)

    tqdm.write("Running validation...")
    # Validation
    y_pred = []
    y_true = []

    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            y_true.extend(local_labels.numpy().tolist())
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            output = model.forward(local_batch)
            val_loss = criterion(output, local_labels)
            output = model.softmax(output)
            output = torch.max(output, 1)[1]
            val_acc = pred_acc(local_labels.cpu(), output.cpu())
            writer.add_scalar('Validation/Accuracy', val_acc, global_val_step)
            writer.add_scalar('Validation/Loss', val_loss, global_val_step)
            global_val_step += params['batch_size']
            y_pred.extend(output.cpu().numpy().tolist())

    tqdm.write(classification_report(y_true=y_true, y_pred=y_pred, target_names=["No Findings", "Pneumonia"]))

    torch.save(model.state_dict(), os.path.join(model_dir, '{0}-epoch-{1}.pt'.format(today.strftime("%b%d_%H-%M-%S"), epoch)))
    if(use_cuda):
        model.cuda()
