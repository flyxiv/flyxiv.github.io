---
title: <Analytics> Starting Spark Session in Azure Synapse Analytics
date: 2025-03-30T08:47:20Z
lastmod: 2025-03-30T08:47:20Z
author: Jun Yeop(Johnny) Na
avatar: /images/favicon.svg
# authorlink: https://author.site
cover: synapse.png
categories:
  - computer vision
tags:
  - data analytics
  - azure
  - synapse analytics
  - geospatial data
  - geosql
# nolastmod: true
draft: false
---

I was doing a project helping an environmental organization find ways to better store and utilize their geospatial data(mostly .shp files and point clouds).

As a solution we recommended moving their data from Dropbox to Azure (Blob Storage) for more flexible storage management and options to analyze data.

To give a proof-of-concept about the benefits of using cloud storage, I made a geospatial analytics prototype using Azure Synapse Analytics.

![Azure Synapse Analytics](./synapse.png)

# 1. Creating Synapse Analytics Workspace

To use Synapse Analytics, we have to create a workspace to use our data. Create new account/file system name

![create workspace](./synapse2.png)

After we create workspace, we'll have it on the workspace menu on Azure Synapse Analytics

![synapse](./synapse3.png)

Create an `apache spark pool`

![pool](./synapse8.png)

Go back to the workspace and click `Open Synapse Studio`

![open synapse studio](./synapse4.png)

Go to develop -> + -> new notebook

![notebook](./synapse6.png)

Rename notebook and assign the spark pool we created earlier

![notebook2](./synapse8.png)

- Now we can use spark resources to do spark tasks in Azure Synapse Analytics.
- Unlike other spark environments, you don't have to start a spark session at the beginning of the notebook, because the notebook automatically starts session in variable 'spark' when the notebook starts.
