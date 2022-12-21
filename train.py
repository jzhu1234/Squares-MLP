import torch
from mlp_model import MultiLayerPerceptron
from mlp_dataloader import SquareDataset, DataLoader
import json, os
from tqdm import tqdm
from torchmetrics import F1Score


if __name__ == '__main__':
    print(torch.cuda.is_available())
    train_data = SquareDataset("C:/Users/zhujo/OneDrive/Documents/squares/train")
    val_data = SquareDataset("C:/Users/zhujo/OneDrive/Documents/squares/val")

    config_path = "mlp_config2.json"
    config = json.loads(open(config_path, 'r').read())

    save_dir = config["save_dir"]
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    train_loader = DataLoader(train_data, **config["loader"])
    val_loader = DataLoader(val_data, **config["loader"])

    model = MultiLayerPerceptron(**config["model"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    metrics = F1Score(task="multiclass", num_classes=3)


    # Softmax
    if config["model"]["activation"] == "softmax":
        loss_fcn = torch.nn.CrossEntropyLoss()
    else:
        loss_fcn = torch.nn.BCEWithLogitsLoss()

    best_f1score = 0
    for epoch in range(config["num_epochs"]):
        model.train(True)
        avg_tloss = 0
        for data, labels in tqdm(train_loader, "Training"):
            optimizer.zero_grad()

            outputs = model(data)

            loss = loss_fcn(outputs, torch.argmax(labels, dim=1))
            loss.backward()
            avg_tloss += loss.item()
            optimizer.step()
        avg_tloss /= len(train_loader)
        # Validation
        # model.train(False)
        model.eval()

        all_pred_lbls = []
        all_true_lbls = []
        for vdata, vlabels in tqdm(val_loader, "Validating"):
            voutputs = model(vdata, train=False)
            pred_lbls = torch.argmax(voutputs, dim=1)
            true_lbls = torch.argmax(vlabels, dim=1)
            all_pred_lbls.append(pred_lbls)
            all_true_lbls.append(true_lbls)
        pred_lbls = torch.cat(all_pred_lbls)
        true_lbls = torch.cat(all_true_lbls)
        f1score = metrics(pred_lbls, true_lbls)

        print()
        print('LOSS train {}'.format(avg_tloss))
        print('Val F1 score {}'.format(f1score))

        # Track best performance, and save the model's state
        if best_f1score < f1score:
            best_f1score = f1score
            model_path = 'model_{}_{:.2f}.h5'.format(epoch, f1score)
            save_path = os.path.join(save_dir, model_path)
            torch.save(model.state_dict(), save_path)
