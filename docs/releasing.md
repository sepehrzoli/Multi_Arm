# Publishing a library release

The package is installed as `best-arm` and imported as `best_arm`.

Update the version in `pyproject.toml` and `src/best_arm/__init__.py`, add release
notes to `CHANGELOG.md`, run the development checks, and commit the changes.
Create an annotated `v<version>` tag and push it. `release.yml` checks the tag,
builds source/wheel distributions, validates their metadata, tests the installed
wheel, and publishes the files in a GitHub release.

## One-time PyPI setup

A maintainer of the existing `best-arm` PyPI project must open
[Publishing settings](https://pypi.org/manage/project/best-arm/settings/publishing/)
and add a GitHub Actions Trusted Publisher with these exact values:

| Field | Value |
| --- | --- |
| Owner | `sepehrzoli` |
| Repository | `Multi_Arm` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

This follows [PyPI's Trusted Publisher setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
It requires no API token in this repository.

## Publish to PyPI

Once the publisher is configured, open **Actions → Publish to PyPI → Run workflow**
in GitHub and enter the existing release tag, such as `v0.1.2`. Run the workflow
from the default branch. The workflow builds and tests the tagged source before
uploading through PyPI's official publishing action.

Confirm the new version on [PyPI](https://pypi.org/project/best-arm/) and verify
it in a fresh environment:

```bash
python -m pip install --upgrade best-arm
python -c "import best_arm; print(best_arm.__version__)"
```

PyPI release versions cannot be overwritten. Fix a published release by bumping
the version and publishing a new one.
