# Release Notes

This page provides an overview of releases and how to work with different versions of PyPSA-China (PIK).

## Current Release

The current stable release is **v1.3.2**. See the [CHANGELOG](https://github.com/pik-piam/PyPSA-China-PIK/blob/main/CHANGELOG.md) for detailed information about changes in this and previous versions.

## Documentation

You can view documentation for different versions using the version selector in the navigation bar. Available versions:

- **stable** - The most recent stable release (points to latest version tag, e.g., v1.3.2)
- **latest** - Latest development version from the develop branch
- **vX.Y.Z** - Specific release versions (e.g., v1.3.2)

## Release Schedule

PyPSA-China (PIK) follows semantic versioning:

- **Major releases** (X.0.0) - Significant changes, may include breaking changes
- **Minor releases** (X.Y.0) - New features, backward compatible
- **Patch releases** (X.Y.Z) - Bug fixes, backward compatible

## Installation of Specific Versions

### Latest Release

```bash
# Clone the repository
git clone https://github.com/pik-piam/PyPSA-China-PIK.git
cd PyPSA-China-PIK
git checkout v1.3.2
```

### Development Version

```bash
# Clone the repository
git clone https://github.com/pik-piam/PyPSA-China-PIK.git
cd PyPSA-China-PIK
# Stay on main branch for latest development version
```

### Specific Version

```bash
# Clone the repository
git clone https://github.com/pik-piam/PyPSA-China-PIK.git
cd PyPSA-China-PIK
git checkout vX.Y.Z  # Replace with desired version
```

## Upgrading

When upgrading between versions, please review the CHANGELOG for:

1. **Breaking changes** - May require updates to your configuration or scripts
2. **New features** - Opportunities to enhance your workflows
3. **Bug fixes** - Issues that have been resolved
4. **Deprecations** - Features that will be removed in future versions

### Upgrade Checklist

- [ ] Review CHANGELOG for your target version
- [ ] Backup your current work and results
- [ ] Update your environment: `conda env update -f environment.yaml`
- [ ] Test with a small example before running large studies
- [ ] Update configuration files if needed
- [ ] Check for deprecated features in your code

## Getting Help

If you encounter issues after upgrading or have questions about a specific release:

1. Check the [documentation](https://pik-piam.github.io/PyPSA-China-PIK/)
2. Review [closed issues](https://github.com/pik-piam/PyPSA-China-PIK/issues?q=is%3Aissue+is%3Aclosed) for similar problems
3. Open a new [issue](https://github.com/pik-piam/PyPSA-China-PIK/issues/new/choose)

## Release History

For the complete release history and detailed change information, see the [CHANGELOG](https://github.com/pik-piam/PyPSA-China-PIK/blob/main/CHANGELOG.md).

### Version 1.3.2 (2025-12-02)

First official release with:
- Versioned documentation support
- Comprehensive documentation
- Automated release workflow
- Official tagging system
- Model description submitted to JOSS Paper


## Staying Updated

To stay informed about new releases:

1. **Watch the repository** on GitHub for release notifications
2. **Star the repository** to show support and get updates
3. **Check the CHANGELOG** periodically for upcoming features in Unreleased section

## Contributing to Releases

Interested in contributing to the next release? See our [Contributing Guide](https://github.com/pik-piam/PyPSA-China-PIK/blob/main/CONTRIBUTING.md) for information on:

- How to propose features
- Development workflow
- Testing requirements
- Documentation standards
