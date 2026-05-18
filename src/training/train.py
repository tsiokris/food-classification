import torch


def train(model, dataloader, epochs, loss_fn, optimizer, device):

    for epoch in range(epochs):
        model.train()
        avg_loss = 0
        correct = 0
        total = 0

        for step, batch in enumerate(dataloader, 1):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            correct += (predictions.argmax(dim=1) == labels).sum().item()
            total += len(labels)

            if step % 10 == 0 or step == len(dataloader):
                print(f"  Epoch {epoch+1}/{epochs} — step {step}/{len(dataloader)} — batch loss: {loss.item():.4f}")

        avg_loss = avg_loss / len(dataloader)
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f} — accuracy: {accuracy:.4f}")


def validate(model, dataloader, loss_fn, device):
    model.eval()
    avg_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            avg_loss += loss.item()
            correct += (predictions.argmax(dim=1) == labels).sum().item()
            total += len(labels)

    accuracy = correct/total
    avg_loss = avg_loss/len(dataloader)
    print(f"loss: {avg_loss} - Accuracy: {accuracy}")



