# PyPSA-China (PIK) Release Preparation Summary

## Overview

This document provides a complete summary of the release infrastructure that has been set up for PyPSA-China (PIK) version 1.3.2.

## Files Created/Modified

### Core Release Files

1. **CHANGELOG.md** - Tracks all changes across versions
   - Follows [Keep a Changelog](https://keepachangelog.com/) format
   - Organized by version with dated releases
   - Categories: Added, Changed, Deprecated, Removed, Fixed, Security

2. **CITATION.cff** - Machine-readable citation file
   - GitHub-recognized citation format
   - Includes software citation and related papers
   - Enables "Cite this repository" feature on GitHub

3. **CITATION.md** - Human-readable citation guide
   - BibTeX entries for software and related papers
   - Instructions for proper citation

4. **CONTRIBUTING.md** - Contributor guidelines
   - Development workflow
   - Coding standards
   - Testing requirements
   - Release process overview

### GitHub Workflows

5. **.github/workflows/docs.yml** - Updated documentation deployment
   - Uses `mike` for versioned documentation
   - Automatically deploys on push to main
   - Maintains `dev` and `latest` versions

6. **.github/workflows/release.yml** - Automated release workflow
   - Triggered by version tags (v*.*.*)
   - Creates GitHub releases automatically
   - Extracts release notes from CHANGELOG.md
   - Deploys versioned documentation

7. **.github/workflows/create-tag.yml** - Tag creation workflow
   - Manual workflow for creating release tags
   - Validates version format
   - Checks version consistency
   - Verifies CHANGELOG entry exists

### GitHub Templates

8. **.github/PULL_REQUEST_TEMPLATE.md** - PR template
   - Ensures comprehensive PR descriptions
   - Checklist for contributors

9. **.github/ISSUE_TEMPLATE/bug_report.md** - Bug report template
   - Structured bug reporting
   - Environment information capture

10. **.github/ISSUE_TEMPLATE/feature_request.md** - Feature request template
    - Structured feature proposals
    - Use case documentation

### Documentation

11. **docs/releases.md** - Release information page
    - Version history
    - Installation instructions for specific versions
    - Upgrade guide

12. **docs/release-guide.md** - Maintainer's release guide
    - Step-by-step release process
    - Pre-release checklist
    - Troubleshooting guide

13. **README.md** - Updated with badges and links
    - Release badges
    - Quick links to documentation
    - Improved structure

## Release Workflow

### For Users

Users can:
1. View versioned documentation at https://pik-piam.github.io/PyPSA-China-PIK/
2. Select specific versions using the version selector
3. Download specific releases from GitHub
4. Track changes via CHANGELOG.md
5. Cite the software using CITATION.cff or CITATION.md

### For Contributors

Contributors should:
1. Follow CONTRIBUTING.md guidelines
2. Update CHANGELOG.md under `[Unreleased]` section
3. Use PR template when submitting changes
4. Follow semantic versioning principles

### For Maintainers

To create a release:

1. **Prepare Release**
   ```bash
   # Update version in workflow/__init__.py
   __version__ = "1.3.0"
   
   # Update CHANGELOG.md with release date
   ## [1.3.0] - 2025-12-02
   ```

2. **Commit Changes**
   ```bash
   git add workflow/__init__.py CHANGELOG.md
   git commit -m "chore: prepare release v1.3.0"
   git push origin main
   ```

3. **Create Tag via GitHub Actions** (Recommended)
   - Go to Actions → "Create Release Tag"
   - Enter version (e.g., `1.3.0`)
   - Click "Run workflow"
   
   OR manually:
   ```bash
   git tag -a v1.3.0 -m "Release version 1.3.0"
   git push origin v1.3.0
   ```

4. **Automated Process**
   - GitHub Release created automatically
   - Documentation deployed with version
   - Release notes extracted from CHANGELOG

## Documentation Versioning

### Mike Integration

The documentation uses `mike` for version management:

- **latest** - Always points to the most recent stable release
- **dev** - Current development version (main branch)
- **vX.Y.Z** - Specific version documentation

### Version Deployment

- Development docs (main branch) deploy as `dev` and `latest`
- Release tags deploy as specific version and update `latest`
- All versions accessible via version selector in docs

## Semantic Versioning

The project follows [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** (X.0.0) - Incompatible API changes
- **MINOR** (X.Y.0) - New features, backward compatible
- **PATCH** (X.Y.Z) - Bug fixes, backward compatible

Current version: **1.3.0**

## GitHub Features Enabled

### Badges

README now includes:
- License badge
- Documentation badge
- Release version badge

### Citation

GitHub's "Cite this repository" feature enabled via CITATION.cff

### Issue Templates

Structured templates for:
- Bug reports
- Feature requests
- (Can add more as needed)

### Pull Request Template

Standardized PR descriptions with checklist

## Next Steps for First Official Release

### Immediate Actions

1. **Review and Merge**: Review all created files and merge to main branch

2. **Create First Tag**: Use the create-tag workflow to create v1.3.0
   ```
   Actions → Create Release Tag → Run workflow
   Version: 1.3.0
   Create Release: Yes
   ```

3. **Verify Deployment**:
   - Check GitHub release page
   - Verify documentation at https://pik-piam.github.io/PyPSA-China-PIK/
   - Test version selector

### Optional Enhancements

4. **Zenodo Integration** (Recommended)
   - Link repository to Zenodo for DOI generation
   - Each release gets automatic DOI
   - Update CITATION.md with DOI once available

5. **GitHub Pages Configuration**
   - Ensure gh-pages branch is the source for GitHub Pages
   - Verify custom domain if applicable

6. **Branch Protection** (Recommended)
   - Protect main branch
   - Require PR reviews
   - Require status checks to pass

7. **Release Announcement**
   - Consider posting about the release
   - Update external documentation
   - Notify users/collaborators

## Testing the Release Infrastructure

Before the first official release, test:

1. **Documentation Build**
   ```bash
   mkdocs build
   mkdocs serve
   ```

2. **Mike Version Management** (local)
   ```bash
   mike deploy 1.3.0-test --push
   mike list
   mike delete 1.3.0-test --push
   ```

3. **Workflow Validation**
   - Create a test tag on a branch
   - Verify workflows run correctly
   - Delete test tag afterwards

## Maintenance

### Regular Tasks

- Update CHANGELOG.md with each PR
- Review and update documentation
- Monitor issues and PRs
- Plan releases based on accumulated changes

### Before Each Release

- Run full test suite
- Update dependencies if needed
- Review open issues for inclusion
- Update documentation
- Prepare release notes

## Additional Resources

- [Keep a Changelog](https://keepachangelog.com/)
- [Semantic Versioning](https://semver.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Mike Documentation](https://github.com/jimporter/mike)
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
- [Citation File Format](https://citation-file-format.github.io/)

## Support

For questions about the release process:
- See docs/release-guide.md
- Open a GitHub Discussion
- Contact repository maintainers

---

**Status**: Ready for first official release v1.3.0

**Created**: 2025-12-02

**Last Updated**: 2025-12-02
