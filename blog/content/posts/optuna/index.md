---
title: <Deep Learning Intermediate> Optuna for Effective Hyperparameter Tuning
date: 2025-05-04T19:47:20Z
lastmod: 2025-05-04T19:47:20Z
author: Jun Yeop(Johnny) Na
avatar: /images/favicon.svg
# authorlink: https://author.site
cover: algorithms.png
categories:
  - Machine Learning
tags:
  - Optuna
# nolastmod: true
draft: false
---

Hyperparameter tuning is an important part of training ML models. Machine learning engineers used to tune hyperparameters by hand or use Grid Search.
These days there are more powerful tools that can help tune hyperparameters in less time.

We're going to look into a easy-to-use tool called `Optuna`

# Problems With Grid Search

1. Takes too long: Search space gets huge as more hyperparameters get inside grid.

```
If there is 86,000 grids already, adding one hyperparameter with just binary options doubles
the number of grids to 172,000
```

2. Doesn't utilize previous search result: As we do the search, we can get a sense of **which range of hyperparameters are more effective to experiment**
3. Only supports discrete grids

ex) After doing 10 experiments with learning rate between 0.01 and 0.0001, we find out that **model diverges with learning rate higher than 0.005-> we can utilize that experiment to limit our search space to learning rate < 0.005 to do more effective searches.**

However grid search doesn't utilize these findings and blindly iterates through all possible combinations.

Optuna solves this problem by

1. **dynamically adjusting their search using previous search results**
2. Early stopping runs that have terrible start.

# Using optuna

We can put optuna in our model using simple steps:

1. Install `optuna` library

```sh
$ pip install optuna
```

2. Wrap our training loop into `objective(trial)` function, and make the function return the metric/loss we want the model to optimize on.

```py
def objective(trial):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(SPRITE_IMG_SIZE),
    ])

    train_dataset, val_dataset, test_dataset = SpriteClassificationDataTable(args.data_dir, transform).split_to_train_val_test()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = torchvision.models.efficientnet_b6(weights=True)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(SPRITE_IDS))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in tqdm(range(args.epochs)):
        model.train()
        running_loss = 0.0

        for (images, labels) in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(loss)

        model.eval()

        false_positive_per_class = torch.zeros(len(SPRITE_IDS))
        false_negative_per_class = torch.zeros(len(SPRITE_IDS))
        true_positive_per_class = torch.zeros(len(SPRITE_IDS))

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, pred_classes = outputs.max(1)

                true_positive_class_indices = torch.where(pred_classes == labels, pred_classes, -1)
                false_positive_class_indices = torch.where(pred_classes != labels, pred_classes, -1)
                false_negative_class_indices = torch.where(pred_classes != labels, labels, -1)

                true_positive_class_indices = true_positive_class_indices[true_positive_class_indices != -1].to('cpu')
                false_positive_class_indices = false_positive_class_indices[false_positive_class_indices != -1].to('cpu')
                false_negative_class_indices = false_negative_class_indices[false_negative_class_indices != -1].to('cpu')

                true_positive_per_class[true_positive_class_indices] += 1
                false_positive_per_class[false_positive_class_indices] += 1
                false_negative_per_class[false_negative_class_indices] += 1


        print(f"Epoch {epoch + 1}, Precision: {true_positive_per_class.sum() / (true_positive_per_class + false_positive_per_class).sum()}")
        print(f"Epoch {epoch + 1}, Recall: {true_positive_per_class.sum() / (true_positive_per_class.sum() + false_negative_per_class.sum())}")

```

3. Give hyperparameter options to the model

```py
import optuna
from optuna.trial import TrialState

	# categorical hyperparameter
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

	# numerical hyperparameter
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
```

4. After every epoch, return if search should end prunely. Return objective function at the end.

```py
        if trial.should_prune():
            raise optuna.TrialPruned()

		precision = true_positive_per_class.sum() / (true_positive_per_class + false_positive_per_class).sum()
        recall = true_positive_per_class.sum() / (true_positive_per_class.sum() + false_negative_per_class.sum())

        trial.report(precision + recall, epoch)

    return precision + recall
```

5. `optuna.create_study`, `study.optimize`, `study.best_trial`

```py
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(complete_trials))
    print("  Number of pruned trials: ", len(pruned_trials))


    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
```

final code:

```py
import torch
import torch.nn as nn
import torchvision
import argparse
import optuna
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from model.sprite_classification.dataset import SpriteClassificationDataTable
from util.sprite_classifications import SPRITE_IDS
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

SPRITE_IMG_SIZE = (80, 40)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    return parser.parse_args()

def objective(trial):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(SPRITE_IMG_SIZE),
    ])

    train_dataset, val_dataset, test_dataset = SpriteClassificationDataTable(args.data_dir, transform).split_to_train_val_test()

    batch_size = trial.suggest_categorical('batch_size', [1, 2, 4, 8,16, 32, 64, 128])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = torchvision.models.efficientnet_b6(weights=True)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(SPRITE_IDS))
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'SGD', 'Adam'])
    lr = trial.suggest_float('lr', 1e-5, 0.01, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    for epoch in tqdm(range(args.epochs)):
        model.train()
        running_loss = 0.0

        for (images, labels) in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()

        if trial.should_prune():
            raise optuna.TrialPruned()

        false_positive_per_class = torch.zeros(len(SPRITE_IDS))
        false_negative_per_class = torch.zeros(len(SPRITE_IDS))
        true_positive_per_class = torch.zeros(len(SPRITE_IDS))

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, pred_classes = outputs.max(1)

                true_positive_class_indices = torch.where(pred_classes == labels, pred_classes, -1)
                false_positive_class_indices = torch.where(pred_classes != labels, pred_classes, -1)
                false_negative_class_indices = torch.where(pred_classes != labels, labels, -1)

                true_positive_class_indices = true_positive_class_indices[true_positive_class_indices != -1].to('cpu')
                false_positive_class_indices = false_positive_class_indices[false_positive_class_indices != -1].to('cpu')
                false_negative_class_indices = false_negative_class_indices[false_negative_class_indices != -1].to('cpu')

                true_positive_per_class[true_positive_class_indices] += 1
                false_positive_per_class[false_positive_class_indices] += 1
                false_negative_per_class[false_negative_class_indices] += 1

        precision = true_positive_per_class.sum() / (true_positive_per_class + false_positive_per_class).sum()
        recall = true_positive_per_class.sum() / (true_positive_per_class.sum() + false_negative_per_class.sum())

        trial.report(precision + recall, epoch)

    return precision + recall

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(complete_trials))
    print("  Number of pruned trials: ", len(pruned_trials))

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
```

# Optuna Algorithms

There are a lot of sampling algorithms used in optuna(Gaussian, TPE, ...). The image below shows a guideline for choosing which algorithm to sample with.

![algorithm](./algorithms.png)
