---
title: Trying Out Google Cloud CLI
date: 2025-08-03T12:47:20Z
lastmod: 2025-08-03T12:47:20Z
author: Jun Yeop(Johnny) Na
avatar: /images/favicon.svg
# authorlink: https://author.site
cover: cover.png
categories:
  - DevOps
  - GCP
  - IaC
tags:
  - DevOps
  - GCP
  - IaC

# nolastmod: true
draft: false
---

Recording commands I can use in gcloud CLI so that I can operate in the cloud with my local command line.

# Setting Config For Project

```sh
gcloud auth application-default set-quota-project {project-name}
gcloud config set project {project-name}
```

# Login to Compute Engine From Local CLI SSH

```sh
gcloud compute ssh {username}@{compute_engine_instance_name} --zone={zone} --command "cd root"
```
