---
title: <Pytorch> Pytorch Optimizations
date: 2025-04-29T10:47:20Z
lastmod: 2025-04-29T10:47:20Z
author: Jun Yeop(Johnny) Na
avatar: /images/favicon.svg
# authorlink: https://author.site
cover: cpp.png
categories:
  - Machine Learning
tags:
  - optimization
  - accelerate
  - pytorch
# nolastmod: true
draft: true
---

# accelerate

using huggingface's `accelerate` helps achieve faster performance with minimum code changes.

- no accelerate:

```python
def pretrain_rtmdet_backbone(model, train_loader, valid_loader, epochs, lr, device):
    optimizer = AdamW(model.parameters(), lr=lr)

    if device == 'cuda':
        model = model.to(device)

    for epoch in range(epochs):
        for batch_idx, (image, target) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            images = image.to(device)

            cls_scores_batch = model(images)

            labels = torch.tensor([label['labels'] for label in target], device=device)
            loss = calculate_classification_loss(cls_scores_batch, labels)

            loss.backward()
            optimizer.step()

            if batch_idx % PERFORMACE_OUTPUT_STEPS == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

        # evaluate model every epoch
        evaluate_model(model, valid_loader, device)


    model.save_backbone(args.save_dir, args.model_name)
```

- with accelerate:

```python
# added
from accelerate import Accelerator

def pretrain_rtmdet_backbone(model, train_loader, valid_loader, epochs, lr, device):
    """Pre-train RTMDET backbone classifier
    """

	# added
	accelerator = Accelerator()
    optimizer = AdamW(model.parameters(), lr=lr)

    if device == 'cuda':
        model = model.to(device)

	# added
	model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    for epoch in range(epochs):
        for batch_idx, (image, target) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            images = image.to(device)

            cls_scores_batch = model(images)

            labels = torch.tensor([label['labels'] for label in target], device=device)
            loss = calculate_classification_loss(cls_scores_batch, labels)

			# added
            accelerator.backward(loss)
            optimizer.step()

            if batch_idx % PERFORMACE_OUTPUT_STEPS == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss}")

        # evaluate model every epoch
        evaluate_model(model, valid_loader, device)


    model.save_backbone(args.save_dir, args.model_name)
```
