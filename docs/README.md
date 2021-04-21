### Overview. 

This uses the `sphinx` framework with the help of `nbsphinx` for rendering `ipynb` files. 

The user-facing documentation is hosted for free on `aiqc.readthedocs.io`. The admin portal is `readthedocs.org/projects/aiqc/`.

---

### Dependencies. 

- Pandoc CLI: https://pandoc.org/installing.html
- Python packages in `docs/requirements.txt`.
    - These are used by ReadTheDocs to install dependencies.
    - You'll notice that everything in `docs/requirements.txt` also appears in `aiqc/requirements_dev.txt`.


---

### Configuration.

These are the important files.

- `docs/conf.py`: settings and extensions.
- `docs/index.rst`: table of contents and homepage.
- `docs/.readthedocs.yml`: environment for ReadTheDocs.
- `docs/requirements.txt`: uses by the yml file.
- Everything else was boilerplate from when sphinx created the project.

---

### Building the docs.
After you make changes to the documentation files, you need to *build* the html pages.

```bash
cd docs
make html
```

You can preview the changes locally by opening the files in `aiqc/docs/_build/html` with a browser.

Alternatively, you can use `pip install sphinx-autobuild` to watch the files for changes and automatically build, but I've never done this.

---

### Publishing the docs.

ReadTheDocs is watching the AIQC GitHub repo for changes pushed to `/docs`:

- `readthedocs.org/accounts/social/connections/`
- `https://readthedocs.org/dashboard/aiqc/integrations/`
- `https://github.com/aiqc/AIQC/settings/hooks`

---

### Quirks.

- Don't forget to run `make html` if you want your changes to show up in the final documentation.
- When adding/ removing/ renaming files to the toctrees, I have to run `make html` twice: once with with and without the `html_sidebars` line of `conf.py` uncommented and then again with it commented.
- Due to JS dependencies, readthedocs.io is not rendering the plots anymore. So I stored them in `/docs/images` and reference them from the notebooks.
- When building, files get replicated. So if you change the name of files in `/docs`, `docs/notebooks`, `docs/images` - then the old files will need to be deleted from `/_build/html`.
- If you make new notebooks from within this folder e.g. `Untitled.ipynb` don't forget to delete them.