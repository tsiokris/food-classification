import torch


def train(model, dataloader, loss_fn, optimizer, device):
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
            print(f"  step {step}/{len(dataloader)} — batch loss: {loss.item():.4f}")

    return avg_loss / len(dataloader), correct / total


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

    return avg_loss / len(dataloader), correct / total
