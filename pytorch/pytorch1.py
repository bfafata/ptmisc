from __future__ import print_function
import torch
#tensors
a = torch.empty(5, 3) #unitialized 5x3 matrix
b = torch.rand(5,3) # randomly initialized matrix
c = torch.zeros(5,3) #zero matrix
d = torch.zeros(5,3, dtype= torch.long) #data type, long = 64 bit integer
e = torch.tensor([4.9, 3]) #construct a tensor directly
a.size() #size of a, outputs a tuple
#operations
a+b
torch.add(c,d) #standard adition
b.add_(torch.rand(5,3)) #any operation proceeded with an underscore changes the tensor it acts upon, like +=

f= torch.randn(4,4) # f.size() = torch.Size([4.4])
g = f.view(16) #reshaping tensors, g.size() = torch.Size([16])
h = f.view(-1,8) # h.size() = torch.Size([2,8]), not 8,2 look at neg

