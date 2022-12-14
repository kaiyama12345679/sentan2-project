import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as plt




class Network(nn.Module):
    
    def __init__(self, num_features) -> None:
        super().__init__()
        self.affine1 = nn.Linear(42, 512)
        self.affine2 = nn.Linear(512, 256)
        self.affine3 = nn.Linear(256, num_features)

    def forward(self, x):
        y = F.relu(self.affine1(x))
        y = F.relu(self.affine2(y))
        y = F.softmax(self.affine3(y))
        return y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("f1")
    parser.add_argument("f2")
    parser.add_argument("f3")
    parser.add_argument("f4")
    args = parser.parse_args()
    num_features = 4

    df1 = pd.read_csv(args.f1)
    df1["label"] = 0
    df2 = pd.read_csv(args.f2)
    df2["label"] = 1
    df3 = pd.read_csv(args.f3)
    df3["label"] = 2
    df4 = pd.read_csv(args.f4)
    df4["label"] = 3

    merged_df = pd.concat([df1, df2, df3, df4], axis=0, ignore_index=True)


    X = torch.FloatTensor(merged_df.drop("label", axis=1).values)
    Y = torch.LongTensor(merged_df["label"].values)
    label = torch.eye(num_features)[Y]
    Dataset = torch.utils.data.TensorDataset(X, label)

    model = Network(num_features)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    acc = []
    num_epochs = 250
    for i in range(num_epochs):
        train_load = torch.utils.data.DataLoader(Dataset, batch_size = 8, shuffle=True)
        correct, sum = 0, 0
        for x, t in train_load:
            optimizer.zero_grad()
            y = model.forward(x)
            predicted = torch.argmax(y, dim=1)
            ans_indices = torch.argmax(t, dim=1)
            correct += torch.sum(predicted == ans_indices)
            sum += x.shape[0]
            loss = criterion(y, t)
            loss.backward()
            optimizer.step()
        acc.append(100 * correct / sum)
        print(f"epoch: {i},   loss: {loss},   accuracy: {100 * correct / sum}%")

    torch.save(model.state_dict(), "models/sample.pth")
    graph_x = range(num_epochs)
    plt.plot(graph_x, acc)
    plt.xlabel("epoch")
    plt.ylabel("accuracy[%]")
    plt.show()
    

