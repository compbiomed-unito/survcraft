# Maintain the documentation

## Build locally

Use Python 3.11 or newer for the documentation toolchain. From the repository
root, install the package and documentation dependencies, then build:

```bash
pip install -e ".[docs]"
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. `-W` makes warnings fail the
build; `--keep-going` reports remaining issues before exiting. The build imports
the actual package, including plotting dependencies, without mocking modules.

## Edit sources

Keep API descriptions in NumPy-style docstrings under `src/survcraft/`. The pages
under `docs/api/` select the classes and methods to include automatically. Write
guides in the Markdown files under `docs/`; MyST supports Sphinx cross-references
and directives within Markdown. Rebuild after changing either code or prose.

Commit documentation sources, configuration, and dependency changes alongside
the code they describe. Generated HTML under `docs/_build/` is ignored by Git.

## GitHub Pages

The `.github/workflows/docs.yml` workflow checks pull requests and builds updates
to the default branch. Only the default branch publishes the resulting HTML to
GitHub Pages. The site has a single current version, with no release-specific
directories or version selector.

For the initial setup, a repository administrator must select **GitHub Actions**
under **Settings → Pages → Build and deployment → Source**. Ensure the
`github-pages` environment permits deployments from the default branch.
After merging the workflow, run **Documentation** from the Actions tab if needed.

The site is published at
<https://compbiomed-unito.github.io/survcraft/>. Pull requests build the
documentation but do not publish it.
