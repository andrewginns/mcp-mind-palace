---
entry_id: uv-create-python-packages
title: Using UV to Create Python Packages from Existing Codebases
tags: [python, uv, packaging, deployment, best-practice]
created: 2025-04-20
last_modified: 2025-04-20
status: active
---

# Using UV to Create Python Packages from Existing Codebases

UV (pronounced "you-vee") is an extremely fast Python package and project manager written in Rust that offers significant improvements over traditional tools like pip, pip-tools, poetry, and virtualenv. One of its powerful capabilities is creating Python packages from existing codebases, which streamlines project organization and deployment.

## Benefits of Using UV for Packaging

- **Speed**: UV performs package operations 10-100x faster than traditional tools
- **Disk Efficiency**: Uses a global cache to minimize redundancy, similar to pnpm in the Node.js ecosystem
- **Cross-Platform**: Consistently works across macOS, Linux, and Windows
- **Editable Installs**: Changes to source code immediately reflect in the installed package
- **Simplified Workflow**: Centralizes project management with one tool instead of multiple specialized ones

## Creating a Package from an Existing Codebase

### Step 1: Install UV

UV can be installed through various methods:

**On macOS/Linux**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows**:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Via pip** (not recommended for long-term use):
```bash
pip install uv
```

### Step 2: Initialize the Project with a Package Structure

Navigate to your existing codebase directory and initialize it as a package:

```bash
cd your-existing-codebase
uv init --package your_package_name
```

This creates a proper package structure and the necessary configuration files:

```
your-existing-codebase/
├── README.md
├── pyproject.toml
└── src/
    └── your_package_name/
        └── __init__.py
```

For library-specific setup with typing support, use the `--lib` flag instead:

```bash
uv init --lib your_package_name
```

### Step 3: Organize Your Existing Code

Move your existing code files into the appropriate package structure under `src/your_package_name/`. This typically involves:

1. Creating appropriate modules (`.py` files) and subpackages (directories with `__init__.py`)
2. Ensuring imports use the new package structure
3. Adding any necessary `__init__.py` files to make Python recognize directories as packages
4. Exposing the relevant functions/classes in the package's main `__init__.py` for clean imports

### Step 4: Configure Dependencies in pyproject.toml

UV automatically creates a `pyproject.toml` file. Update it with the necessary metadata and dependencies:

```toml
[project]
name = "your-package-name"
version = "0.1.0"
description = "Description of your package"
readme = "README.md"
authors = [
    { name = "Your Name", email = "your.email@example.com" }
]
requires-python = ">=3.8"
dependencies = [
    "dependency1>=1.0.0",
    "dependency2==2.1.0",
]

[project.scripts]
your-command = "your_package_name:main_function"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

You can add dependencies using UV's command:

```bash
uv add dependency1 "dependency2==2.1.0"
```

### Step 5: Test the Package Locally

Create and activate a virtual environment to test your package:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install your package in "editable" mode:

```bash
uv sync
```

UV will automatically install your package in development mode, allowing changes to reflect immediately without reinstallation.

### Step 6: Build the Distribution

When ready to distribute your package, build it:

```bash
uv build
```

This creates the package distribution files in the `dist/` directory:
- `.whl` (wheel) file for binary distribution
- `.tar.gz` file for source distribution

### Step 7: Install or Distribute the Package

You can now install the built package in other environments:

```bash
uv pip install dist/your_package_name-0.1.0-py3-none-any.whl
```

## Working with Existing Project Types

### Converting a Poetry Project to UV

If you have an existing Poetry project, UV can work with it directly:

1. Navigate to your Poetry project
2. Initialize UV environments with `uv venv`
3. Synchronize dependencies with `uv sync`

UV understands and respects the Poetry configuration in `pyproject.toml`.

### Converting a pip requirements.txt Project

For projects using `requirements.txt`:

1. Navigate to your project directory
2. Initialize it with UV: `uv init --package your_package_name`
3. Install existing dependencies: `uv pip install -r requirements.txt`
4. Optionally, generate a locked dependency file: `uv pip freeze > requirements-lock.txt`

## Best Practices for Packaging with UV

1. **Use the src/ Directory Layout**: This prevents import confusion and follows modern Python packaging standards
2. **Include Type Hints**: Use `py.typed` and type annotations for better IDE support
3. **Create Comprehensive Documentation**: Include examples in your README.md
4. **Version Carefully**: Follow semantic versioning for your package
5. **Test Package Installation**: Verify the package can be properly installed and used before distribution

## Common Issues and Solutions

### Import Errors After Packaging

If you encounter import errors after packaging:

```python
# Bad - Absolute import from within package
from your_package_name.utils import helper_function

# Good - Relative import for internal package usage
from .utils import helper_function
```

### Missing Files in the Package

By default, only Python files are included. To include other file types, add a `[tool.hatch.build.targets.wheel]` section to your `pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
include = [
    "src/your_package_name/data/*.json",
    "src/your_package_name/templates/*.html",
]
```

### Dependency Conflicts

Use UV's advanced dependency resolution capabilities:

```bash
uv add --extra-constraint "conflicting-package<2.0.0" required-package
```

## Conclusion

UV's streamlined approach to Python packaging provides a powerful way to transform existing codebases into distributable packages. By combining development and packaging workflows into a single, fast, and efficient tool, UV helps maintain clean project organization while simplifying dependency management.

When setting up packages with UV, you benefit from the tool's speed and advanced features while maintaining compatibility with the broader Python ecosystem of tools and repositories.