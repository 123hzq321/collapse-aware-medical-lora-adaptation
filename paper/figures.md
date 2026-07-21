# Figure Drafts

## Figure 1. Experimental Pipeline

```mermaid
flowchart LR
    A["Medical QA datasets<br/>PubMedQA / MedMCQA / MedQA"] --> B["full45k replay mixture<br/>44,978 examples"]
    B --> C["Frozen baseline evaluation<br/>0.5B / 1.5B / 3B"]
    B --> D["LoRA adaptation<br/>seed 42 / 43 / 44"]
    C --> E["Collapse-aware BENCHMARKER"]
    D --> E
    E --> F["Accuracy and macro-F1"]
    E --> G["Dominant-rate, entropy,<br/>missing labels, collapse flag"]
    E --> H["Paired McNemar tests"]
    F --> I["Paper tables"]
    G --> I
    H --> I
```

## Figure 2. Collapse Mitigation Mechanism

```mermaid
flowchart LR
    subgraph Before["Before adaptation"]
        Q1["Medical QA prompt"] --> M0["Frozen small LM"]
        M0 --> P0["Label-prior / format bias"]
        P0 --> C0["Answer collapse<br/>high dominant-rate<br/>low entropy<br/>low macro-F1"]
    end

    subgraph Update["Task-driven LoRA adaptation"]
        D["Multi-task medical QA replay"] --> L["LoRA weight update"]
        L --> R["Answer-space recalibration"]
    end

    subgraph After["After adaptation"]
        Q2["Medical QA prompt"] --> M1["Updated LM"]
        R --> M1
        M1 --> P1["More balanced predictions"]
        P1 --> C1["Collapse removed<br/>dominant-rate down<br/>entropy up<br/>macro-F1 up"]
    end
```

## Visual Emphasis for Final Artwork

- Figure 1 should be clean and pipeline-oriented: data, update, benchmarker, statistics, tables.
- Figure 2 should be mechanism-oriented: frozen bias, LoRA update, answer-space recalibration.
- Use three metric arrows in Figure 2: dominant-rate down, entropy up, macro-F1 up.
- Avoid implying autonomous self-training. Label the mechanism as task-driven LoRA adaptation.
