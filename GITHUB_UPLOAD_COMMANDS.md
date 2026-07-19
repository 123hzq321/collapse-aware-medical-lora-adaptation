# GitHub Upload Commands

Suggested repository name:

`collapse-aware-medical-self-update`

Suggested public URL:

`https://github.com/123hzq321/collapse-aware-medical-self-update`

## Option A: Create and Push with GitHub CLI

Run from this folder:

```powershell
git init
git branch -M main
git add .
git commit -m "Initial release for collapse-aware medical self-update"
gh repo create 123hzq321/collapse-aware-medical-self-update --public --source . --remote origin --push
```

## Option B: If the Empty GitHub Repository Already Exists

```powershell
git init
git branch -M main
git add .
git commit -m "Initial release for collapse-aware medical self-update"
git remote add origin https://github.com/123hzq321/collapse-aware-medical-self-update.git
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
- per-example prediction JSONL files
