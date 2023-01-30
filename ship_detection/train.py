import torch


# from loguru import logger
Tensorboard - 
MLFlow & ClearML
NeptuneML - 
Weights&Biases - 

loader = train_loader 
# model = nn.Module()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.losses._Loss


def train_one_epoch(model, loader, optimizer, **kwargs):
    pass

def validate(model, loader):
    pass

def train():

    train_one_epoch()
    validate()


# for epoch in range(num_epochs)
#     for idx, (x, y) in enumerate(loader):
#         optimizer.zero_grad()
#         prediction = model(x)
#         loss = criterion(prediction, y)
#         loss.backward() # scalar 
#         # logger.info(loss)
#         # logger.warning(loss)
#         logger.info(loss)
#         optimizer.step()

train, val

# Как понять что модель учиться?
1. Лосс снижается
2. Можешь заоверфитить 1 батч (лосс ~0)
3. Лосс на валидации тоже падает

x = torch.randn(10).requires_grad()

print(x.requires_grad)


