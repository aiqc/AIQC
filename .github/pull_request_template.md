# Pull Request (PR) Template

---

> *Not sure how to make a PR? Watch this [video](https://www.youtube.com/watch?v=rgbCcBNZcdQ) (fork it, clone it, branch it, code it, push it, PR it). If you are forking it, then you'll also want to [set an `upstream` repo](https://www.atlassian.com/git/tutorials/git-forks-and-upstreams ) to keep your fork up to date.*

---

## Does this solve an existing [Issue](https://github.com/aiqc/aiqc/issues)? 
- If so, please link to the issue.
- If not, please describe *what* your change does and *why*.

---

> *Put an `x` inside the applicable checkboxes `[ ]`.*

## Primary type of change:
- [ ] New feature.
- [ ] Bug fix.
- [ ] Refactor.
- [ ] Documentation.
- [ ] Build/ devops.
- [ ] Test.
- [ ] Other.

## Checklist:
- [ ] I have actually ran this code locally to make sure it works.
- [ ] I have pulled and merged the latest `main` into my feature branch.
- [ ] I have ran the existing [tests](https://github.com/aiqc/aiqc/new/main/.github#how-to-run-tests), and inspected areas that my code touches.
- [ ] I have updated the tests to include my changes.
- [ ] I have included updates to the documentation.
- [ ] I have built the documentation files `make html`.

> Describe the env you tested on: OS, Python version, shell type (e.g. cli, ide, notebook).

## Contains breaking changes:
- [ ] Yes (explain what functionality it breaks).
- [ ] No

---

## How to run the tests.
Reference `https://github.com/aiqc/AIQC/blob/main/aiqc/tests/tests.ipynb` for a notebook that can be used to run the tests. Please make a copy of this file and don't push it to the repo.

The source code for tests is located in `aiqc/aiqc/tests/__init__.py`. The `run_jobs()` is called manually because we often need to inspect the `Queue` object.
