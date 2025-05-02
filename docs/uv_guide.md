# UV Package Management Guide

## Initial Setup

1. **Install UV** (if not already installed):
```bash
# Unix-like systems
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile "install.ps1"
.\install.ps1
Remove-Item "install.ps1"
```

2. **Create Virtual Environment**:
```bash
uv venv .venv

# Activate the environment
# Unix-like systems
source .venv/bin/activate

# Windows
.\.venv\Scripts\Activate.ps1
```

## Managing Dependencies

### Initial Setup with requirements.in

1. **Create requirements.in** with your direct dependencies:
```bash
# Example requirements.in
streamlit
pymupdf
minio
sentence-transformers
langchain
transformers
torch
python-dotenv
tqdm
langchain-core
langchain-nvidia-ai-endpoints
langchain-community
faiss-cpu
langfuse
ragas
datasets
numpy
pandas
```

2. **Compile requirements.txt**:
```bash
uv pip compile requirements.in -o requirements.txt
```

3. **Sync your environment**:
```bash
uv pip sync requirements.txt
```

### Adding a New Package

1. **Add the package to requirements.in**:
```bash
# Add your new package to requirements.in
echo "new-package" >> requirements.in
```

2. **Recompile requirements.txt**:
```bash
uv pip compile requirements.in -o requirements.txt
```

3. **Sync your environment**:
```bash
uv pip sync requirements.txt
```

### Development Dependencies

1. **Create requirements-dev.in**:
```bash
-r requirements.in
pytest
pytest-cov
black
ruff
mypy
```

2. **Compile development requirements**:
```bash
uv pip compile requirements-dev.in -o requirements-dev.txt
```

3. **Sync development environment**:
```bash
uv pip sync requirements-dev.txt
```

## Common Commands

### Compile Options
```bash
# Basic compile
uv pip compile requirements.in -o requirements.txt

# Compile with specific Python version
uv pip compile --python 3.9 requirements.in -o requirements.txt

# Compile with platform-specific packages
uv pip compile --platform win_amd64 requirements.in -o requirements.txt

# Compile with specific index
uv pip compile --index-url https://pypi.org/simple/ requirements.in -o requirements.txt
```

### Sync Options
```bash
# Basic sync
uv pip sync requirements.txt

# Sync with specific index
uv pip sync --index-url https://pypi.org/simple/ requirements.txt

# Sync with no cache
uv pip sync --no-cache-dir requirements.txt
```

## Workflow Examples

### Adding a New Package
```bash
# 1. Add to requirements.in
echo "new-package==1.0.0" >> requirements.in

# 2. Compile
uv pip compile requirements.in -o requirements.txt

# 3. Sync
uv pip sync requirements.txt
```

### Updating All Packages
```bash
# 1. Compile with --upgrade
uv pip compile --upgrade requirements.in -o requirements.txt

# 2. Sync
uv pip sync requirements.txt
```

### Installing a Package Temporarily
```bash
# Install without modifying requirements
uv pip install package-name

# To remove it later
uv pip uninstall package-name
```

## Best Practices

1. **Always use requirements.in** for direct dependencies
2. **Keep requirements.txt** for locked versions
3. **Use requirements-dev.in** for development tools
4. **Commit both .in and .txt files** to version control
5. **Use specific versions** in requirements.in when needed
6. **Regularly update** dependencies with `--upgrade`
7. **Use sync** instead of install to maintain consistency

## Troubleshooting

1. **If sync fails**:
```bash
# Try with --no-cache-dir
uv pip sync --no-cache-dir requirements.txt
```

2. **If compile fails**:
```bash
# Try with verbose output
uv pip compile -v requirements.in -o requirements.txt
```

3. **If package not found**:
```bash
# Check available versions
uv pip index versions package-name
```

4. **If version conflict**:
```bash
# Try with --no-deps
uv pip compile --no-deps requirements.in -o requirements.txt
``` 