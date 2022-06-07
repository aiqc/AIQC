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
- Everything else in the root directory (e.g. 'make.bat' or 'Makefile') was boilerplate from when sphinx created the project. Don't delete them.


#### Static assets.
- `html_static_path` in conf
- Register each asset file `html_css_files` & `html_js_files` in conf.py
- Register `/_static/fonts` within css `@font-face`.
- `make html` is supposed to replicate to `.../docs/_build/html/_static/css/custom.css` 
  but it seem I have to manually overwrite the _build css file at that location.


#### Build process.
After you make changes to the documentation files, you need to *build* the html pages.

```bash
cd docs
make html
```

You can preview the changes locally by opening the files in `aiqc/docs/_build/html` with a browser.

Alternatively, you can use `pip install sphinx-autobuild` to watch the files for changes and automatically build, but I've never done this.

When adding/removing/renaming files to the toctrees, I have to run `make html` twice: once with with and without the `html_sidebars` line of `conf.py` uncommented and then again with it commented.

When changing file names/ paths, things duplicated and orphaned in `/_build/html`.

```bash
cd docs
make clean
```

If the build fails remotely, then check for dependency issues here <https://github.com/spatialaudio/nbsphinx/issues> e.g. maintainer fixed a bug where it was not working with Jinja2(v3).

There were too many standalone notebook pages so I put them in 'gallery.rst' and now the "WARNING: document isn't included in any toctree" is to be expected.


#### Automated publishing.

ReadTheDocs is watching the AIQC GitHub repo for changes pushed to `/docs`:

- `readthedocs.org/accounts/social/connections/`
- `https://readthedocs.org/dashboard/aiqc/integrations/`
- `https://github.com/aiqc/AIQC/settings/hooks`


#### Redirects

Only run when a page 404s

- https://docs.readthedocs.io/en/stable/user-defined-redirects.html


#### Images

- Used this site for favicons <https://icoconvert.com/> `.ico` format.
- If you use the sphinx directive `.. image:: some_pic.png` then it needs a local path.
- When referencing images in markdown `![some_img]path.png` the underscore breaks it on RTD.
- Due to JS dependencies, readthedocs.io is not rendering the plots anymore. So I stored them in `/docs/images` and reference them from the notebooks.


#### Links
For notebook hyperlinks, I seem to have to use a `[](.html)` syntax to get it to work on RTD.
https://nbsphinx.readthedocs.io/en/0.8.1/markdown-cells.html#Links-to-Other-Notebooks


#### Plotly Plots

Plotly plots stopped working because they could no longer access JS dependencies

Here is some of the troubleshooting I tried:
Adding this options due to plotly browser error w require.min.js
https://cdnjs.com/libraries/require.js/2.1.10    Integrity in the "</> Copy Script Tag" button.
https://github.com/readthedocs/sphinx_rtd_theme/issues/788#issuecomment-772785054 also had to save `https://cdn.plot.ly/plotly-latest.min.js` to `/Users/layne/Desktop/aiqc/docs/_build/html/notebooks/plotly.js`

```js
nbsphinx_requirejs_options = {
 	"src": "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js",
 	"integrity": "sha512-VCK7oF67GXNc+J7zsu5o57jtxhLA75nSMHGaq8Q8TCOxDj4nMDw5dhQZvm9Cd9RN+3zgcodqbKcRc9gyPP8a2w==",
 	"crossorigin": "anonymous"
}
```