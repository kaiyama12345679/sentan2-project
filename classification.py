import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("f1")
parser.add_argument("f2")
parser.add_argument("f3")
parser.add_argument("f4")
args = parser.parse_args()

df1 = pd.read_csv(args.f1)
df1["label"] = 0
df2 = pd.read_csv(args.f2)
df2["label"] = 1
df3 = pd.read_csv(args.f3)
df3["label"] = 2
df4 = pd.read_csv(args.f4)
df4["label"] = 3

merged_df = pd.DataFrame([df1, df2, df3, df4])

X = torch.FloatTensor(merged_df.drop("label", axis=1).values)
Y = torch.LongTensor(merged_df["label"].values)

Dataset = torch.utils.data.TensorDataset(X, Y)


class Network(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.affine1 = nn.Linear(20, 512)
        self.affine2 = nn.Linear(512, 256)
        self.affine3 = nn.Linear(256, 4)

    def forward(self, points):
        if len(points) < 20:
            return None
        y = F.relu(self.affine1(points))
        y = F.relu(self.affine2(y))
        y = F.softmax(self.affine3(y))
        return y


model = Network()
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
for i in range(10):
    train_load = torch.utils.DataLoader(Dataset, batch_size = 4, shuffle=True)
    for x, t in train_load:
        optimizer.zero_grad()

        y = model.forward(x)
        loss = criterion(y, t)
        loss.backward()

    print(f"epoch: {i},   loss: {loss}")
        


    

