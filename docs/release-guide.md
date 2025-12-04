# Release Guide for PyPSA-China (PIK)

This guide is for maintainers who are preparing and publishing releases.

## Pre-Release Checklist

Before creating a new release, ensure the following:

- [ ] All planned features and bug fixes are merged
- [ ] Tests are passing on the main branch
- [ ] Documentation is up to date
- [ ] Version number is decided (following semantic versioning)

## Semantic Versioning

This project follows [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** (X.0.0): Incompatible API changes or major restructuring
- **MINOR** (X.Y.0): New features, backward-compatible
- **PATCH** (X.Y.Z): Bug fixes, backward-compatible

## Release Process

### 1. Update Version Number

Edit `workflow/__init__.py`:

```python
__version__ = "X.Y.Z"  # Update to new version
```

### 2. Update CHANGELOG.md

Update the CHANGELOG with the new version:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- List new features

### Changed
- List changes to existing functionality

### Fixed
- List bug fixes

### Deprecated
- List deprecated features

### Removed
- List removed features
```

Add the version link at the bottom:
```markdown
[X.Y.Z]: https://github.com/pik-piam/PyPSA-China-PIK/releases/tag/vX.Y.Z
```

### 3. Commit Changes

```bash
git add workflow/__init__.py CHANGELOG.md
git commit -m "chore: prepare release vX.Y.Z"
git push origin main
```

### 4. Create Release Tag

#### Option A: Using GitHub Actions (Recommended)

1. Go to the repository on GitHub
2. Click on "Actions"
3. Select "Create Release Tag" workflow
4. Click "Run workflow"
5. Enter the version number (e.g., `1.3.0` without the 'v' prefix)
6. Choose whether to create the GitHub release immediately
7. Click "Run workflow"

This will:
- Validate the version format
- Check if the tag already exists
- Verify the version matches `workflow/__init__.py`
- Verify CHANGELOG.md has an entry for this version
- Create and push the tag
- Optionally trigger the release workflow

#### Option B: Manual Tag Creation

```bash
# Create an annotated tag
git tag -a vX.Y.Z -m "Release version X.Y.Z"

# Push the tag to GitHub
git push origin vX.Y.Z
```

### 5. Automated Release Process

Once the tag is pushed, the Release workflow automatically:

1. **Creates a GitHub Release**:
   - Extracts release notes from CHANGELOG.md
   - Creates the release with appropriate metadata
   - Marks it as the latest release

2. **Deploys Versioned Documentation**:
   - Builds documentation with MkDocs
   - Deploys to GitHub Pages using mike
   - Updates version aliases (vX.Y.Z and latest)
   - Sets the new version as default

### 6. Verify the Release

After the workflows complete:

1. **Check the GitHub Release**:
   - Visit https://github.com/pik-piam/PyPSA-China-PIK/releases
   - Verify the release appears with correct version and notes

2. **Check the Documentation**:
   - Visit https://pik-piam.github.io/PyPSA-China-PIK/
   - Verify the version selector shows the new version
   - Check that content is correct for the new version

3. **Verify Version Links**:
   - Ensure all links in CHANGELOG.md work correctly
   - Test the documentation version switcher

### 7. Post-Release

1. **Announce the Release**:
   - Consider posting about the release to relevant channels
   - Update any external documentation or websites

2. **Prepare for Next Development Cycle**:
   - Consider updating version to next development version (e.g., X.Y.Z+1-dev)
   - Add new `[Unreleased]` section to CHANGELOG.md

## Hotfix Releases

For urgent bug fixes on a released version:

1. Create a hotfix or patch branch from the release tag:
   ```bash
   git checkout -b hotfix/vX.Y.Z+1 vX.Y.Z
   ```

2. Make the necessary fixes

3. Update version to X.Y.Z+1 in `workflow/__init__.py`

4. Update CHANGELOG.md with the patch notes

5. Commit and create a PR to main

6. After merging, follow the normal release process for the patch version

## Pre-releases and Release Candidates

For testing before a major release:

1. Create a pre-release version: `vX.Y.Z-rc1`

2. Mark the GitHub release as a "pre-release" (checkbox option)

3. Deploy documentation to a separate version:
   ```bash
   mike deploy X.Y.Z-rc1 --push
   ```

## Troubleshooting

### Tag Already Exists

If you need to move a tag:

```bash
# Delete local tag
git tag -d vX.Y.Z

# Delete remote tag
git push origin :refs/tags/vX.Y.Z

# Create new tag
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z
```

**Warning**: Only do this if the release hasn't been widely distributed.

### Release Workflow Failed

1. Check the Actions tab for error details
2. Fix the issue (usually in CHANGELOG.md or version files)
3. You can re-run the workflow from the Actions tab
4. Or manually trigger using workflow_dispatch

### Documentation Not Deploying

1. Ensure mike is installed with version >=2.0.0
2. Check that gh-pages branch exists
3. Verify GitHub Pages is enabled in repository settings
4. Check workflow logs for specific errors

## Version Management with Mike

Mike maintains versioned documentation in the `gh-pages` branch.

### Useful Mike Commands

```bash
# List all deployed versions
mike list

# Set a version as default
mike set-default VERSION --push

# Delete a version
mike delete VERSION --push

# Deploy without updating latest alias
mike deploy VERSION --push

# Deploy and update latest alias
mike deploy --update-aliases VERSION latest --push
```

### Local Testing

Test documentation builds locally before releasing:

```bash
# Build and serve locally
mkdocs serve

# Build with mike locally (without pushing)
mike deploy VERSION

# Serve with mike's version selector
mike serve
```

## Release Checklist Template

Copy this checklist for each release:

```markdown
## Release vX.Y.Z Checklist

- [ ] Tests passing on main
- [ ] Documentation updated
- [ ] Version updated in workflow/__init__.py
- [ ] CHANGELOG.md updated with version and date
- [ ] Changes committed and pushed to main
- [ ] Tag created (via GitHub Actions or manually)
- [ ] Release workflow completed successfully
- [ ] GitHub release created and verified
- [ ] Documentation deployed and accessible
- [ ] Version selector shows new version
- [ ] Announcement prepared (if applicable)
- [ ] Next development cycle prepared
```

## Questions?

If you have questions about the release process, please:
- Review this guide thoroughly
- Check previous releases for examples
- Contact the repository maintainers
- Open a discussion on GitHub
