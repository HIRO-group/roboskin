# [PACKAGE NAME]

# Installation
```
python setup.py install
```

# Run Examples
Examples are in `./examples/`
For example, Run
```
python examples/01_add_numbers.py
```


# Documentations
You can find the documentation 

# For Developers
## Test
```
python setup.py test
```

# Contribution guide
New features and bug fixes are welcome. Send PRs.
This project is using GitHub flow ([See here for details](https://guides.github.com/introduction/flow/)) for development so do not try to push directly to master branch (It will be rejected anyway)

## Target python versions
Target python versions are 2.7, 3.4, 3.5 and 3.6 (as of August 2018).
Use six, [future](https://pypi.org/project/future/) or any other libraries to keep compatibility among above python versions.

## Repository structure
### Module Structure
Write your features under ./robotic_skin/
Write your tests for the features under ./tests/


## Write TESTS 
When adding new feature such as function/class, always and must write test(s) unless it will be rejected.

### Where should I write my feature's tests?
When writing tests, for example for feature_module.py, please create test module file of name test_feature_module_name.py and place exactly at the same layer of your feature module.
See below. <br>

```
├── robotic_skin 
│   ├── __init__.py
│   ...
│   ├── your_awesome_module.py
...
└── tests
    ├── __init__.py
    ...
    ├── test_your_awesome_module.py
    │
    ...
```

## Write documents
Write documents of your new function/class/feature and explain what it does.
Writing documents is hard but helps others understanding what you implemented and how to use it.

### Style
We use numpy style docstring. When writing the docs, follow numpy style.
[See here](https://numpydoc.readthedocs.io/en/latest/) for details. 
