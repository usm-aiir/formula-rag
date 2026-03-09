# Formula-Aware Multimodal RAG for Mathematics

A multimodal RAG system for mathematics that treats **formulas as a first-class modality**, alongside text and images.

## Overview

Mathematical meaning is distributed across:
- **Natural language** — explanations, definitions
- **Symbolic expressions** — formulas with structure that matters
- **Visual content** — diagrams, plots, screenshots

This project explores whether retrieval and generation improve when formulas are modeled structurally (syntax trees, operator graphs) rather than as plain text.

## Project Structure

See [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) for the full directory layout.

## Dataset & Benchmark

See [`docs/DATASET_DESIGN.md`](docs/DATASET_DESIGN.md) for:
- Data source (Math Stack Exchange)
- Inclusion criteria
- Multimodal post schema
- Annotation layers
- Benchmark tasks (retrieval, grounding, generation)

## Phases

1. Literature review and scope refinement
2. Dataset collection and filtering
3. Baseline retrieval pipeline
4. Formula-aware retrieval module
5. Multimodal fusion and generation
6. Evaluation and analysis

## License

TBD
