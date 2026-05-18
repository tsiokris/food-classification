import torch


def train(model, dataloader, epochs, loss_fn, optimizer):
 
    for epoch in range(epochs):
        model.train()
        avg_loss = 0

        for batch in dataloader:
            images, labels = batch
            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

        avg_loss = avg_loss/len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} — loss: {avg_loss}")


def validate(model, dataloader, loss_fn):
    model.eval()
    avg_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            avg_loss += loss.item()
            correct += (predictions.argmax(dim=1) == labels).sum().item()
            total += len(labels)

    accuracy = correct/total
    avg_loss = avg_loss/len(dataloader)
    print(f"loss: {avg_loss} - Accuracy: {accuracy}")



