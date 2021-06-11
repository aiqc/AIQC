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
The source code for tests is located in `aiqc/aiqc/tests/__init__.py`. The `run_jobs()` is called manually because we often need to inspect the `Queue` object.

Once we are confident that our schema handles all data types and analysis types, we will invest more rigorous testing.
```python
import aiqc
from aiqc import tests

q1 = tests.make_test_queue('keras_binary')
q1.run_jobs()

q2 = tests.make_test_queue('keras_multiclass')
q2.run_jobs()

q3 = tests.make_test_queue('keras_regression')
q3.run_jobs()

q4 = tests.make_test_queue('keras_text_binary')
q4.run_jobs()

q5 = tests.make_test_queue('keras_image_binary')
q5.run_jobs()

q6 = tests.make_test_queue('keras_sequence_binary')
q6.run_jobs()

q7 = tests.make_test_queue('pytorch_binary')
q7.run_jobs()

q8 = tests.make_test_queue('pytorch_multiclass')
q8.run_jobs()

q9 = tests.make_test_queue('pytorch_regression')
q9.run_jobs()

q10 = tests.make_test_queue('pytorch_image_binary')
q10.run_jobs()
```
