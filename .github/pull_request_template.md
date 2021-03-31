# Pull Request (PR) Template

> *Not sure how to make a PR? Watch this [video](https://www.youtube.com/watch?v=rgbCcBNZcdQ).
> Fork it. Clone it. **Branch it.** Code it. Push it. PR it.
> You'll also want to [set an `upstream` repo](https://www.atlassian.com/git/tutorials/git-forks-and-upstreams ) to keep your fork up to date.*

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
- [ ] I have ran the existing [tests](https://github.com/aiqc/aiqc/new/main/.github#how-to-run-tests), and inspected areas that my code touches.
- [ ] I have updated the tests to include my changes.
- [ ] I have included updates to the documentation.
- [ ] I am pushing a *branch*, not my *main/ master*.

## Contains breaking changes:
- [ ] Yes (explain what functionality it breaks).
- [ ] No

---

## How to run tests.
The source code for tests is located in `aiqc/aiqc/tests/__init__.py`. 

Once we are confident that our schema handles all data types and analysis types, we will invest more rigorous testing.
```python
import aiqc
from aiqc import tests

b1 = tests.make_test_batch('binary')
b1.run_jobs()

b2 = tests.make_test_batch('multiclass')
b2.run_jobs()

b3 = tests.make_test_batch('regression')
b3.run_jobs()

b4 = tests.make_test_batch('image_binary')
b4.run.jobs()
```
