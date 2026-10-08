# ManiMux documentation

The [guide](index.md) is the documentation website's entry point. These files have
separate audiences and responsibilities:

- Root `README.md`: a concise project overview, demo and quick-start links.
- `.agents/skills/*/SKILL.md`: instructions that route an agent through development or setup.
- `docs/`: user guides, shared repository conventions and integration protocols.
- `docs/mkdocs.yml`, `docs/requirements.txt`, `docs/_build/`: website build configuration.
- `licenses/`: upstream license texts and third-party attribution.

Skills link to the relevant protocol pages rather than duplicating their specifications.
README stays short; it does not embed the skills or the full protocols. Update a protocol
in its owning Markdown page and rebuild the website.

## Preview locally

From the repository root, in a documentation-only environment:

```bash
uv venv .venv-docs --python 3.12
uv pip install --python .venv-docs/bin/python -r docs/requirements.txt
.venv-docs/bin/mkdocs serve -f docs/mkdocs.yml --dev-addr 127.0.0.1:8000
```

Build a static site with `.venv-docs/bin/mkdocs build --strict -f docs/mkdocs.yml`. Output goes to the
ignored `site/` directory. Serve that directory on any static host. Repository
links use `main`; update `edit_uri` in
`docs/mkdocs.yml` and `REVISION` in `docs/_build/hooks.py` together when publishing
from another branch.

## Publishing

The [public guide](https://sii-liulab.github.io/manimux/) is deployed by
`.github/workflows/docs.yml` after documentation changes are pushed to `main`.
The workflow builds with strict link validation and publishes only `site/` to GitHub
Pages. It does not install robot/model environments or initialize submodules.
The repository Pages setting must use **GitHub Actions**, and the `github-pages`
environment must permit deployments from `main`. When changing the publishing
branch, update that environment rule, the workflow trigger and the source-link settings
above together.

Keep onboarding in `usage/`, extension contracts in `development/`, supported
launch recipes in `deployment/`, and detailed algorithms in `advanced/`. Team
study decisions/results are maintained in `docs/experiments.md`. The register is
kept in the repository but excluded from the public documentation build. It is
not required for everyday inference; preserve existing decisions and results.
Historical development logs remain available in Git history rather than the guide.
