from mnistdl import *


#nueral network from scratch

#using pytorch tensor operations
#not slick, but this is how they did it in like 2017 lul

weights = torch.randn(784, 10) / math.sqrt(784) #784=28*28 + xavier initlisation 1/sqrt(n)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x): #activation function
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias) # @ is dot product

bs=64 #batch size
xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions
yb = y_train[0:bs]
#implimenting neg log likilihood loss function
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

#accuracy function, not strictly needed
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

#from IPython.core.debugger import set_trace

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for
for epoch in range(epochs):
        for i in range((n - 1) // bs + 1): #select mini batch
            #         set_trace()
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]#grabbing batch
            yb = y_train[start_i:end_i]
            pred = model(xb) #use model to make predictions
            loss = loss_func(pred, yb)#calculate loss

            loss.backward()#update the gradiants of the model
            with torch.no_grad():
                weights -= weights.grad * lr
                bias -= bias.grad * lr
                weights.grad.zero_()
                bias.grad.zero_()

print(loss_func(model(xb), yb), accuracy(model(xb), yb))