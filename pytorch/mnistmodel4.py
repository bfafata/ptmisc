#refactoring using optim
from mnistdl import *
from torch import nn
import torch.nn.functional as F
from torch import optim
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

loss_func = F.cross_entropy
bs=64
xb = x_train[0:bs]
yb = y_train[0:bs]
lr=.5
epochs=2

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10) #here

    def forward(self, xb):
        return self.lin(xb)
def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()
print(loss_func(model(xb), yb),accuracy(model(xb),yb))

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb),accuracy(model(xb), yb))