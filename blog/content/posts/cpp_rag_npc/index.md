---
title: <C++> Implementing RAG Controlled Game NPC in Unreal
date: 2025-04-14T18:33:20Z
lastmod: 2025-04-14T18:33:20Z
author: Jun Yeop(Johnny) Na
avatar: /images/favicon.svg
# authorlink: https://author.site
cover: cpp.png
categories:
  - gaming
  - ML
  - Unreal
tags:
  - cpp
  - voice recognition
  - RAG
# nolastmod: true
draft: false
---

To control NPCs with natural language we need to define more Unreal Components that will streamline LLM access in an Unreal Engine environment.

We need:
1) Unreal character class that has components needed for controlling via LLM.
2) A request handler class that handles requesting to online LLM servers and parsing their response.

# Define Plugin "LlmNpc"

Create "LlmNpc" Folder in Project's `Plugins` directory.

