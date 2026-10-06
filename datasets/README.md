# Dataset provenance and licenses

The repository-level MIT License applies only to original source code authored
for this project. It does not relicense third-party benchmark data or artifacts
derived from them. Each benchmark and its generated ambiguous variant remain
subject to the terms of the corresponding source dataset.

## Spider 1.0

- Source: [Yale Spider project](https://yale-lily.github.io/spider)
- License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Citation: Tao Yu et al., *Spider: A Large-Scale Human-Labeled Dataset for
  Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*, EMNLP 2018.

The project reformats Spider metadata, generates retrieval documents, and
creates content-partitioned ambiguous variants.

## BIRD

- Source: [BIRD project](https://bird-bench.github.io/)
- License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Citation: Jinyang Li et al., *Can LLM Already Serve as a Database Interface?
  A Big Bench for Large-Scale Database Grounded Text-to-SQLs*, NeurIPS,
  volume 36.

The project reformats BIRD metadata, generates retrieval documents, and creates
content-partitioned ambiguous variants.

## BEAVER

- Sources: [beaver-query](https://huggingface.co/datasets/beaverbench/beaver-query)
  and [beaver-table](https://huggingface.co/datasets/beaverbench/beaver-table)
- License declared by both source dataset cards: MIT
- Citation: Peter Baile Chen et al., *BEAVER: An Enterprise Benchmark for
  Text-to-SQL*, 2024.

The project converts the published database dumps to SQLite and transforms the
queries and schemas into its common benchmark representation. Access to the
table source remains subject to the conditions of its gated distribution.

## ARCHER

- Source and citation: Danna Zheng, Mirella Lapata, and Jeff Z. Pan,
  [*Archer: A Human-Labeled Text-to-SQL Dataset with Arithmetic, Commonsense
  and Hypothetical Reasoning*](https://aclanthology.org/2024.eacl-long.6/),
  EACL 2024.

This repository does not assert an additional license for ARCHER. Consult the
terms supplied with the original dataset before redistributing its data or
project-generated derivatives.

## SQaLe3

- Sources: [queries_new3](https://huggingface.co/datasets/cwolff/queries_new3)
  and [schemas_new3](https://huggingface.co/datasets/cwolff/schemas_new3)
- Source revisions: `7039ddd42ca37dfee6388d02ddcb65fdf120541d` and
  `25ec7203c9035b675cd97de5d06484f0b7bde58a`, respectively

The build retains verified query--schema pairs, materializes SQLite databases,
and validates query results. This repository does not assert an additional
license for SQaLe3; consult the source repositories before redistributing the
data or project-generated derivatives.
