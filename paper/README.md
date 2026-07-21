# Paper Draft

This folder contains an English arXiv/workshop-style manuscript draft.

Files:

- `main.tex`: main manuscript.
- `references.bib`: BibTeX references.
- `figures/`: compiled vector PDF figures used in the manuscript.
- `figures.md`: earlier Mermaid drafts kept as planning notes.

Suggested compile command:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Current framing:

> Collapse-aware evaluation of task-driven LoRA adaptation for resource-limited medical language models.

The paper intentionally avoids claiming autonomous self-improvement. It defines the method as lightweight supervised LoRA adaptation and centers the contribution on answer-collapse diagnosis and mitigation.
