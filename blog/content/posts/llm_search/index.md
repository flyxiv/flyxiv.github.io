---
title: <LLM Search> Use of LLM in Search Engines
date: 2025-06-27T10:03:19Z
lastmod: 2025-06-27T10:04:20Z
author: Jun Yeop(Johnny) Na
avatar: /images/favicon.svg
# authorlink: https://author.site
cover: fine-tuning.png
categories:
  - LLM
  - Supervised fine tuning
tags:
  - python
  - PEFT
  - QLoRA
  - Huggingface
  - Pytorch

# nolastmod: true
draft: false
---

# Before LLM

- Keyword/Token-based relevance calculation

[before](./cover.png)

# After LLM

- RAG retrieval-based relevance calculation
- **The order of documents retrieved by the RAG system matters!!! -> Reranking is also done by LLMs**

[after](./new_search.png)

There are four parts that affect the performance of the search engine:

- Embedding (Used for calculatingm relevance)
- RAG system
- Fine-tuned reranking and response model.

# Embedding

## Embedding Methods

There are many ways to

- word-based embedding: does well on semantic understanding, but hard to learn symbolistic information(ex) hard to train similarity between 'run' and 'running'), can fall into OOV
- sub-word embedding: use sub-words: longer than character, but smaller than word(prefix, postfix, character sequence, ...) - best of both worlds
- letter-based embedding: no OOV, too long sequence, hard to train semantics.

**sub-word embedding is currently the go-to standard for LLM tokens**

## Types of Sub-Word Embedding

1. n-gram(FastText): `ex) n=3: apple = sum(['<ap', 'app', 'ppl', 'ple', 'le>' ])`
2. BPE(Byte Pair Encoding): merging "character sequences" that frequently come after each other. - Used in GPT models
3. WordPiece: Similar to BPE, but **merges based on maximizing likelihood**
   ex) if Word is made of A, B, C, ...

   - Likelihood of whole Word = P(Word) = P(A) _ P(B) _ ...
   - When A, B is merged: P(Word) = P(AB) \* ...
   - so **P(AB) / P(A) \* P(B) is the likelihood increase ratio when merging A and B**
   - Keep merging until wanted size is reached.

4. SentencePiece: Similar but **doesn't divide word by spaces - more robust to non-english**

## Evaluating Embedding

### Intrinsic Evaluation

Evaluates the embedding itself.

- Word Similarity

  - human-labeled similarity score between text pairs ex) (computer, coffee, 0.1), (cat, dog, 0.5)
  - check correlation between the text pairs' cosine similarity and human label
  - has open source datasets: `WordSim-353`, `SimLex-999`, `KorSim(for Korean)`

- Word Analogy
  - Check if the embedding gives good results for semantic arithmetic between texts
  - ex) `vec('Germany') - vec('Berlin') + vec('Paris') = vec('France')

### Extrinsic Evaluation

Evaluates the embedding by using it in specific tasks.

- Sentiment analysis, Classification, Ranking etc.
- In ranking: MRR, MAP, NDCG etc.
- MTEB evaluates the embedding in various different tasks and gives final result

## RAG System

### Naive RAG

- Traditional index(chunking)-retrieve-generation
- Cannot solve hallucination problem if there is no right document
- Might not retrieve the right documents

### Advanced RAG

- Divide retrieval into "Pre/Post" process

#### Pre-Retrieval Process

Process and augment user's quetion into parts for better retrieval.

1. Rewrite user question
2. Divide into smaller parts
3. Add similar keywords

and retrieve documents for each part.

#### Post-Retrieval Process

Process document's order and content so that it can be better recognized by the LLM.

1. Re-ranking
2. Information Compaction

#### Modular RAG

Create RAG pipeline as **replaceable modular parts**

- Hybrid research: semantic + lexical search
- Iterative Retrieval: More than one search
- Self-RAG: Self-critique of its own response
- Adaptive Retrieval: Do more search only when LLM thinks more information is needed

new RAG techniques are being researched very actively. We also need ways to add Media data and there could be better ways to evaluate its performance.

## Fine-Tuned Models

Similar to Embedding: Use Intrinsic/Extrinsic Evaluation

- Intrinsic: BLEU, ROUGE - doesn't take into account semantic meaning
- Extrinsic: Chatbot-arena, MT-Bench, MMLU, HumanEval, LLM as a judge

# LLM Structures

## 1. Encoder-based

- Focused on "understanding meaning of text"
- BERT, RoBERTa
- Trained on:
  1.  MLM(Masked Language Model): Mask random words in a sentence and make LLM fill it.
  2.  NSP(Next Sentence Prediction): Find out if sentence B is next sentence of sentence A in `[CLS] sentence A [SEP] sentence B [SEP]`

## 2. Decoder-based

- Focused on "generating next text"
- GPT, Llama, ...
- Trained on: autoregressive next word prediction

## 3. Transformer-based

- Focused on "recovering damaged text dat"
- Trained on:

  1. T5(Text-to-Text Transfer Transformer)

  - ex) `Thank you for inviting me to your party last week`
  - encoder: `"Thank you <X> me to your party <Y> week."` - masks parts of sentence
  - decoder: `<X> for inviting <Y> last <Z>` - finds out the missing parts

  2. BART (Bidirectional and Auto-Regressive Transformer): damage text in various ways

  - token masking
  - token deletion
  - sentence permutation
  - document rotation

# Reasoning Model

The difference between reasoning model and classic LLM is in **the fine-tuning process**

- COT fine-tuning

  - regular SFT: [question, answer]
  - reason-focused SFT: [question, "reasoning steps", answer]

- RLHF/DPO: evaluate the 'thought process' instead of the answer itself.
  - logicality
  - efficientcy (3 + 5 vs (1 + 1 + 1) + (1 + 1 + 1 + 1 + 1))
  - accuracy

# Various Transformer Variations

## Positional Embedding

- Absolute Positional Embedding
  - Learnable Positional Embedding: learn embedding for each position - can only process fixed length of sentences
  - Sinusoidal: Most common. Can theoretically extrapolate for long sentences, but in production performance degrades when sentences get long.
- Relative Positional Embedding
  - `(QK/sqrt(n) + relative embedding) * value`
  - has embedding table for every relative position value (ex) -2: [-0.3, 0.8, ...], 2: [1.3, 0.4, ...])
  - No extrapolation problem, but more costly
- Hybrid Embeddings
  - Rotary Positional Embedding: rotate token embedding vector in certain degree depending on position. - **change inner product of QK only by changing relative degree, not their length.**
    - rotation is done based on **absolute position, but the attention score only depends on relative position of the two tokens.**
  - Alibi: decrease attention score for far tokens.
