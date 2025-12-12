# Contributing to PyPSA-China (PIK)

Thank you for your interest in contributing to PyPSA-China (PIK)! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Versioning](#versioning)
- [Release Process](#release-process)

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/PyPSA-China-PIK.git
   cd PyPSA-China-PIK
   ```
3. **Set up the development environment** following the [installation guide](https://pik-piam.github.io/PyPSA-China-PIK/installation/quick_start/)
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Reporting Bugs

- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include detailed steps to reproduce the issue
- Provide environment information (OS, Python version, etc.)
- Include relevant error messages and logs

### Suggesting Features

- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Clearly describe the feature and its benefits
- Explain the use case and motivation
- Consider if it aligns with the project's goals

### Code Contributions

1. **Check existing issues** to avoid duplicate work
2. **Discuss major changes** by opening an issue first
3. **Write clear commit messages** following conventional commits:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for test additions/modifications
   - `refactor:` for code refactoring
   - `chore:` for maintenance tasks

4. **Update documentation** to reflect your changes
5. **Add tests** for new functionality
6. **Update CHANGELOG.md** under the `[Unreleased]` section

## Development Workflow

### Setting Up Your Environment

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate pypsa-china

# Install development dependencies
pip install pytest pytest-cov black flake8
```

### Making Changes

1. Make your changes in your feature branch
2. Test your changes locally:
   ```bash
   # Run tests
   pytest tests/

   # Run specific test suite
   pytest tests/unit/
   pytest tests/integration/
   ```

3. Format your code:
   ```bash
   # Format with black (if used in project)
   black workflow/scripts/
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

### Submitting a Pull Request

1. **Ensure pre-commit checks pass** (Important!)
   ```bash
   # Run pre-commit on all files
   pre-commit run --all-files

   # Or run on specific files
   pre-commit run --files path/to/your/files
   ```

   The pre-commit checks will automatically run in CI, but running them locally first saves time.

2. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub:
   - **Target branch**:
     - Use `develop` for new features and non-critical bug fixes
     - Use `main` only for hotfixes or critical patches to the current stable release
   - Most contributions should target the `develop` branch

3. **Fill out the PR template** completely
4. **Link related issues** using keywords like "Closes #123"
5. **Wait for review** and address any feedback
6. **Ensure CI checks pass**

### Branch Strategy

- **develop**: Active development branch for new features and improvements
  - Continuous integration and testing
  - Documentation deployed as "latest" version
  - Merged to `main` before releases

- **main**: Stable branch reflecting the latest release
  - Should always be in a releasable state
  - Tagged with version numbers for releases
  - Documentation deployed as "stable" version

- **feature branches**: For individual features or bug fixes
  - Created from and merged back to `develop`
  - Named descriptively: `feature/description` or `fix/description`

## Coding Standards

### Python Code

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes using Google style with type hints:
  ```python
  def my_function(param1: str, param2: int) -> bool:
      """
      Brief description of function.

      Args:
          param1 (str): Description of param1
          param2 (int): Description of param2

      Returns:
          bool: Description of return value
      """
      pass
  ```

### Snakemake Rules

- Use clear rule names that describe the action
- Document rule purpose with comments
- Specify inputs, outputs, and resources explicitly
- Use `log:` directive to capture outputs

### Configuration Files

- Use YAML format for consistency
- Add comments explaining non-obvious settings
- Validate configurations before use

## Testing

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test names: `test_<functionality>_<scenario>`
- Include both positive and negative test cases

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=workflow --cov-report=html

# Run specific test file
pytest tests/unit/test_specific.py

# Run tests matching a pattern
pytest -k "test_pattern"
```

## Documentation

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

### Documentation Guidelines

- Write clear, concise documentation
- Include code examples where appropriate
- Update tutorial notebooks if adding new features
- Add cross-references to related documentation
- Keep the README.md up to date

## Versioning

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality (backward-compatible)
- **PATCH** version for backward-compatible bug fixes

### Updating Version

When preparing a release, update the version in:

1. `workflow/__init__.py`:
   ```python
   __version__ = "X.Y.Z"
   ```

2. Update `CHANGELOG.md` with the new version and release date

## Release Process

Releases are managed by project maintainers:

1. **Prepare the release:**
   - Update version number in `workflow/__init__.py`
   - Update `CHANGELOG.md` with release date
   - Ensure documentation is current

2. **Create and push a tag:**
   ```bash
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   git push origin vX.Y.Z
   ```

3. **Automated release workflow:**
   - GitHub Actions automatically creates the release
   - Documentation is deployed with versioning via mike
   - Release notes are generated from CHANGELOG.md

4. **Post-release:**
   - Verify release on GitHub
   - Check documentation deployment
   - Announce release to users

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Maintain a professional environment
- Questions are always welcome but please check the documentation first!

## Questions?

- Check the [documentation](https://pik-piam.github.io/PyPSA-China-PIK/)
- Open a [discussion](https://github.com/pik-piam/PyPSA-China-PIK/discussions)
- Contact the maintainers via GitHub issues

## License

By contributing to PyPSA-China (PIK), you agree that your contributions will be licensed under the [MIT License](LICENSES/MIT.txt).

---

Thank you for contributing to PyPSA-China (PIK)! 🎉
