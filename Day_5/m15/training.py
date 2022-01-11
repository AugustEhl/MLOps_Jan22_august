import pickle
import torch
from model import MyAwesomeModel
from torch import nn
from torch import optim
from data import mnist

print("Training day and night")
lr = 0.001
weight_decay = 1e-3
num_epochs = 5
batch_size = 5

# TODO: Implement training loop here
model = MyAwesomeModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr,betas=(0.85,0.89),weight_decay=weight_decay)
train_set = torch.load('data.pth')[0]
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
train_loss = []
train_acc = []
for i in range(num_epochs):
    print('Epoch ',i+1,' of ',num_epochs)
    running_loss = 0
    running_acc = 0
    for images, labels in trainloader:
        labels = labels.long()
        #print(labels.shape)
        optimizer.zero_grad()   
        output = model(images)
        #print(output.shape)
        loss = criterion(output,labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        y_pred = (nn.Softmax(dim=1)(output)).argmax(dim=1)
        running_acc += torch.sum(y_pred == labels).item()/labels.shape[0]
        #print(y_pred.shape)
    train_acc.append(running_acc / len(trainloader))
    train_loss.append(running_loss / len(trainloader))
    print('Train loss: ', running_loss,
    '\ttrain accuracy: ', running_acc/len(trainloader) * 100,'%')
torch.save(model, 'final_model.pth')
pickle.dump(train_loss,open( "train_loss.p", "wb" ))
pickle.dump(train_acc, open('train_acc.p', "wb"))