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
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            avg_loss += loss.item()

            top5_preds = predictions.topk(5, dim=1).indices
            labels_col = labels.unsqueeze(1)
            correct_top1 += (top5_preds[:, :1] == labels_col).any(dim=1).sum().item()
            correct_top3 += (top5_preds[:, :3] == labels_col).any(dim=1).sum().item()
            correct_top5 += (top5_preds[:, :5] == labels_col).any(dim=1).sum().item()
            total += len(labels)

    return (
        avg_loss / len(dataloader),
        correct_top1 / total,
        correct_top3 / total,
        correct_top5 / total,
    )
