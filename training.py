import argparse
import copy
import json

import cv2
import matplotlib.pyplot as plt
import time
import torch

from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from detection.data import get_dataloaders
from detection.evaluation import get_correct_preds, bbox_iou
from detection.models.resnet import ResNet34
from detection.losses import BCELoss, DetectionLoss

data_cat = ["train", "valid"]


def show_predictions(x, predictions, targets, pred_threshold=0.5):
    images = x.detach().cpu()
    preds = predictions.detach().cpu()
    targs = targets.detach().cpu()
    #     f, subs = plt.subplots(1, images.shape[0])
    for idx, img in enumerate(images):
        img = cv2.cvtColor(images[idx][0].numpy(), cv2.COLOR_GRAY2BGR)
        curr_context = plt.gca()  # current pyplot context
        pred = preds[idx]
        target = targs[idx]
        print(f"Target: {target} ; Prediction: {pred}")
        if pred[0] > pred_threshold:
            h, w = pred[-2:] * 256
            x, y = pred[1:-2] * 256
            x -= w / 2
            y -= h / 2
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if target[0] == 1:
            h, w = target[-2:] * 256
            x, y = target[1:-2] * 256
            x -= w / 2
            y -= h / 2
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        plt.imshow(img)
        plt.show()


def train_detection_model(model, criterion, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs,
                          plot_predictions=False, pred_threshold=0.5):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_miou_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_miou = 0.0
    costs = {x: [] for x in data_cat}  # for storing costs per epoch
    metrics = {
        cat: {
            "classification_accuracy": [],
            "mIoU": [],
            "iou": {n: [] for n in range(num_epochs)},
            "loss": []
        } for cat in data_cat
    }
    metrics["best_model_acc"] = best_acc
    metrics["best_model_miou"] = best_miou
    print("Train batches:", len(dataloaders["train"]))
    # print("Valid batches:", len(dataloaders["valid"]), "\n")
    for epoch in tqdm(range(num_epochs)):
        epoch_start = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)
        # Each epoch has a training and validation phase
        for phase in data_cat:
            model.train(phase == "train")
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                print(i, end="\r")
                inputs, target = data
                inputs = Variable(inputs).cuda()
                labels = Variable(target).cuda()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)  # batch_size x target.shape (8x5)
                loss = criterion(outputs, labels).mean()
                running_loss += loss.data.item()  # [0]
                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                # statistics
                correct_preds = get_correct_preds(outputs[:, 0], labels[:, 0])
                running_corrects += correct_preds

                for j in range(labels.shape[0]):
                    if labels[j, 0].item() == 1.0:
                        metrics[phase]["iou"][epoch].append(bbox_iou(outputs[j, 1:].detach(), labels[j, 1:]).item())
                if i % 100 == 0:
                    items_ran = (i + 1) * len(labels)
                    print(f"Running Corrects: {running_corrects} / {items_ran} ({100*(running_corrects / items_ran):0.5f}%)"
                          f"; Running Loss {(running_loss / items_ran):0.5f}")  # 8 being batch size
                if i % 2000 == 0 and plot_predictions:
                    show_predictions(inputs, outputs, labels, pred_threshold)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_miou = torch.Tensor(metrics[phase]["iou"][epoch]).cuda().mean()
            metrics[phase]["mIoU"].append(epoch_miou.data.item())

            metrics[phase]["loss"].append(epoch_loss)
            metrics[phase]["classification_accuracy"].append(epoch_acc)
            print(f"{phase} Loss: {epoch_loss:.10f} Acc: {epoch_acc:.10f} mIOU: {epoch_miou:.10f}")
            # print("Confusion Meter:\n", confusion_matrix[phase].value())
            # deep copy the model
            if phase == "valid":
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), f"checkpoint_{epoch + 1}epochs_acc_{epoch_acc}_miou_{epoch_miou}.pt")
                if epoch_miou > best_miou:
                    best_miou = epoch_miou.data.item()
                    best_miou_model = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), f"checkpoint_{epoch + 1}epochs_acc_{epoch_acc}_miou_{epoch_miou}.pt")

        metrics["best_model_acc"] = best_acc
        metrics["best_model_miou"] = best_miou
        # Timing stuff
        epoch_time = time.time() - epoch_start
        time_elapsed = time.time() - since
        print(f"Epoch {epoch + 1} took {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s")
        print(f"Time elapsed since start: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print()
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best valid Acc: {best_acc:4f}")
    # plot_training(costs, accs)
    # load best model weights
    torch.save(model.state_dict(), "end_of_training.pt")
    model.load_state_dict(best_miou_model)
    return model, metrics


def train_classification_model(model, criterion, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs,
                               plot_predictions=False, pred_threshold=0.5):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    metrics = {
        cat: {
            "classification_accuracy": [],
            "loss": []
        } for cat in data_cat
    }
    metrics["best_model_acc"] = best_acc
    print("Train batches:", len(dataloaders["train"]))
    # print("Valid batches:", len(dataloaders["valid"]), "\n")
    for epoch in tqdm(range(num_epochs)):
        epoch_start = time.time()
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)
        # Each epoch has a training and validation phase
        for phase in data_cat:
            model.train(phase == "train")
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                print(i, end="\r")
                inputs, target = data
                inputs = Variable(inputs).cuda()
                labels = Variable(target).cuda()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)  # batch_size x target.shape (8x5)
                loss = criterion(outputs, labels).mean()
                running_loss += loss.data.item()  # [0]
                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                # statistics
                correct_preds = get_correct_preds(outputs, labels)
                running_corrects += correct_preds

                if i % 100 == 0:
                    items_ran = (i + 1) * len(labels)
                    print(f"Running Corrects: {running_corrects} / {items_ran} ({100*(running_corrects / items_ran):0.5f}%)"
                          f"; Running Loss {(running_loss / items_ran):0.5f}")  # 8 being batch size
                if i % 2000 == 0 and plot_predictions:
                    show_predictions(inputs, outputs, labels, pred_threshold)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            metrics[phase]["loss"].append(epoch_loss)
            metrics[phase]["classification_accuracy"].append(epoch_acc)
            print(f"{phase} Loss: {epoch_loss:.10f} Acc: {epoch_acc:.10f}")
            # print("Confusion Meter:\n", confusion_matrix[phase].value())
            # deep copy the model
            if phase == "valid":
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), f"checkpoint_{epoch + 1}epochs_acc_{epoch_acc}.pt")

        metrics["best_model_acc"] = best_acc
        # Timing stuff
        epoch_time = time.time() - epoch_start
        time_elapsed = time.time() - since
        print(f"Epoch {epoch + 1} took {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s")
        print(f"Time elapsed since start: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print()
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best valid Acc: {best_acc:4f}")
    # plot_training(costs, accs)
    # load best model weights
    torch.save(model.state_dict(), "end_of_training.pt")
    model.load_state_dict(best_model_wts)
    return model, metrics


MODEL_NAME_TO_MODEL_CLASS = {
    "resnet": ResNet34
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_type", help="classification / detection", default="detection")
    parser.add_argument("--model", help="model name", default="resnet")
    parser.add_argument("--num_epochs", help="number of epochs", default=100)
    parser.add_argument("--batch_size", help="batch size", default=8)
    parser.add_argument("-nf", "--normals_fraction", help="batch size", default=0.4)
    parser.add_argument("--pred_threshold", help="batch size", default=0.5)
    parser.add_argument("--output_file", help="output json file", default="training_results.json")

    args = parser.parse_args()
    num_epochs = int(args.num_epochs)
    batch_size = int(args.batch_size)
    normals_fraction = float(args.normals_fraction)
    pred_threshold = float(args.pred_threshold)
    output_file = args.output_file
    training_type = args.training_type.lower()

    assert args.model in MODEL_NAME_TO_MODEL_CLASS, \
        f"Unsupported model name; supported names: {list(MODEL_NAME_TO_MODEL_CLASS.keys())}"
    assert training_type in ["detection", "classification"], f"Unsupported training type {training_type}; " \
                                                             f"please use either 'detection' or 'classification'"

    Model = MODEL_NAME_TO_MODEL_CLASS[args.model]
    dataloaders = get_dataloaders(training_type, normal_fraction=normals_fraction, batch_size=batch_size)

    if training_type == "detection":
        model = Model(num_classes=5)
        criterion = DetectionLoss()
        training_function = train_detection_model
    else:  # training_type == "classification":
        model = Model(num_classes=1)
        dataloaders = get_dataloaders(training_type)
        criterion = BCELoss()
        training_function = train_classification_model
    model.cuda()
    optimizer = Adam(model.parameters(), lr=1e-3)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, verbose=True)
    model, metrics = training_function(model=model, criterion=criterion, optimizer=optimizer, dataloaders=dataloaders,
                                       scheduler=lr_scheduler, num_epochs=num_epochs, pred_threshold=pred_threshold,
                                       dataset_sizes={key: len(dataloader.dataset) for key, dataloader in
                                                      dataloaders.items()}, plot_predictions=False)
    print(f"Saving training metrics to {output_file}")
    print(metrics)
    with open(output_file, "w") as metrics_file:
        json.dump(metrics, metrics_file)


if __name__ == '__main__':
    main()
