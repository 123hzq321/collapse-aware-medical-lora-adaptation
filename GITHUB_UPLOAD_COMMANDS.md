# GitHub Upload Commands

Suggested repository name:

`collapse-aware-medical-lora-adaptation`

Suggested public URL:

`https://github.com/123hzq321/collapse-aware-medical-lora-adaptation`

## Option A: Create and Push with GitHub CLI

Run from this folder:

```powershell
git init
git branch -M main
git add .
git commit -m "Initial release for collapse-aware medical LoRA adaptation"
gh repo create 123hzq321/collapse-aware-medical-lora-adaptation --public --source . --remote origin --push
```

## Option B: If the Empty GitHub Repository Already Exists

```powershell
git init
git branch -M main
git add .
git commit -m "Initial release for collapse-aware medical LoRA adaptation"
git remote add origin https://github.com/123hzq321/collapse-aware-medical-lora-adaptation.git
git push -u origin main
```

## Before Pushing

Check that no large or restricted files are staged:

```powershell
git status --short
git ls-files
```

This repository should not include:

- model checkpoints
- LoRA adapter weights
- `.venv`
- downloaded benchmark datasets
- full downloaded benchmark datasets
