## Overview. 

This uses the `sphinx` framework with the help of `nbsphinx` for rendering `ipynb` files. 

The user-facing documentation is hosted for free on `aiqc.readthedocs.io`. The admin portal is `readthedocs.org/projects/aiqc/`.


#### Running the docs.

Remember that the documentation is user-facing, so, if you want to run the documentation notebooks yourself, be sure to duplicate the `.ipynb` file you are running so that you don't change the final product. If you make new notebooks from within this folder e.g. `Untitled.ipynb`, or anywhere else, don't forget to either delete them or transfer them out before running the build process.

---

## Building the documentation website.

#### Dependencies. 

- Pandoc CLI: https://pandoc.org/installing.html
- Python packages in `docs/requirements.txt`.
    - These are used by ReadTheDocs to install dependencies.
    - You'll notice that everything in `docs/requirements.txt` also appears in `aiqc/requirements_dev.txt`.


#### Configuration files.

These are the important files:

- `docs/conf.py`: settings and extensions.
- `docs/index.rst`: table of contents and homepage.
- `docs/.readthedocs.yml`: environment for ReadTheDocs.
- `docs/requirements.txt`: uses by the yml file.
- Everything else was boilerplate from when sphinx created the project.


#### Build process.
After you make changes to the documentation files, you need to *build* the html pages.

```bash
cd docs
make html
```

You can preview the changes locally by opening the files in `aiqc/docs/_build/html` with a browser.

Alternatively, you can use `pip install sphinx-autobuild` to watch the files for changes and automatically build, but I've never done this.


#### Automated publishing.

ReadTheDocs is watching the AIQC GitHub repo for changes pushed to `/docs`:

- `readthedocs.org/accounts/social/connections/`
- `https://readthedocs.org/dashboard/aiqc/integrations/`
- `https://github.com/aiqc/AIQC/settings/hooks`

---

## Quirks.

- There were too many notebooks so I put them in 'tutorials.rst' and now the "WARNING: document isn't included in any toctree" is to be expected.
- If the build fails remotely, then check for dependency issues here: https://github.com/spatialaudio/nbsphinx/issues
- Don't forget to run `make html` if you want your changes to show up in the final documentation.
- When adding/ removing/ renaming files to the toctrees, I have to run `make html` twice: once with with and without the `html_sidebars` line of `conf.py` uncommented and then again with it commented.
- Due to JS dependencies, readthedocs.io is not rendering the plots anymore. So I stored them in `/docs/images` and reference them from the notebooks.
- When building, files get replicated. So if you change the name of files in `/docs`, `docs/notebooks`, `docs/images` then the old files will need to be deleted from `/_build/html`.
- When I ran into problems with broken image references on ReadTheDocs, I ran `make clean` to delete the `/_build` folder and then `make html`.
- The `nbsphinx==0.8.6` maintainer fixed a bug where it was not working with Jinja2(v3). 
- Don't delete 'make.bat' or 'Makefile'.