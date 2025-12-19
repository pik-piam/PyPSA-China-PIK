# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security

## [1.3.0] - 2025-12-02

### Added
- Official release preparation with versioned documentation
- Comprehensive documentation at https://pik-piam.github.io/PyPSA-China-PIK/
- MkDocs-based documentation with Material theme
- Support for versioned documentation using mike
- GitHub Actions workflow for automated documentation deployment
- Test suite with pytest integration
- Configuration templates for various study types (myopic, REMIND coupling)
- Global Energy Monitor data integration
- Technology configuration system
- Network plotting with customizable styles
- Policies (subsidies) and differentiated fuel costs

### Changed
- Improved documentation structure with tutorials and reference guides
- Enhanced installation instructions including PIK HPC cluster setup
- Updated dependencies and environment management

### Fixed
- Solar bin minimum value corrections
- Land use data processing improvements
- Nature reserves data updates
- Configuration report bug fixes
- Test data and shapes updates

## [1.2.0] - Prior Release

### Added
- Land use and slope data processing
- Protected areas integration
- Enhanced testing framework

## [1.1.x] - Prior Releases

### Added
- Multiple enhancements to the modeling workflow
- Improved data handling and processing

## [1.0.x] - Initial PIK Releases

### Added
- Initial PIK adaptation of PyPSA-China
- Provincial resolution energy system model
- Capacity expansion optimization
- Integration with PyPSA framework
- Snakemake workflow management
- Basic documentation structure

---

## Version History Notes

PyPSA-China (PIK) is based on the paper by Zhou et al, which extends a version original developed by Hailiang Liu et al. This changelog tracks changes from version 1.0.0 onwards in the PIK implementation.

For detailed information about specific changes, see the [commit history](https://github.com/pik-piam/PyPSA-China-PIK/commits/main) on GitHub.

[Unreleased]: https://github.com/pik-piam/PyPSA-China-PIK/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/pik-piam/PyPSA-China-PIK/releases/tag/v1.3.0
[1.2.0]: https://github.com/pik-piam/PyPSA-China-PIK/releases/tag/v1.2.0
