![](https://github.com/HIRO-group/robotic_skin/workflows/Python%20application/badge.svg)

![](https://github.com/HIRO-group/robotic_skin/workflows/Docker%20Image%20CI/badge.svg)

# HIRO Robotic Skin
## Current Release
- `0.0.1` as of 2020/01/14

## Target Python versions
Target Python version is 3.7.4, and a Python version >= 3.6 is required.


# HTTPS Installation
```
pip install --upgrade git+https://github.com/HIRO-group/robotic_skin.git
```

# SSH Installation
```
pip install --upgrade git+ssh://git@github.com/HIRO-group/robotic_skin.git
```

# Run Examples
Examples are in `./examples/`
For example, Run:
```
python examples/01_add_numbers.py
```

or if you have a connected accelerometer (ADXL335):
```
python examples/02_read_adxl335.py
```


# Documentation
You can find the documentation [here](http://hiro-group.ronc.one/robotic_skin/).


# For Developers
New features and bug fixes are welcome. Send PRs. <br>
This project is using GitHub flow ([See here for details](https://guides.github.com/introduction/flow/)) for development, so do not try to push directly to master branch (It will be rejected anyway).


## Contribution Steps
1. Add your feature under `./robotic_skin/`
2. Comment classes and functions with [docstring](https://en.wikipedia.org/wiki/Docstring)
3. Add Example 
4. Write Unit Test under `./tests/`
5. Run Test Locally `python setup.py test`
6. Run Test Remotely (Travis automatically deals this)
7. Write Documentations under `./docsrc/`. See [docsrc/README.md](docsrc/README.md). 

## Test [MUST]
When adding new feature such as `function`/`class`, you always and must write test(s) unless it will be rejected. <br>
Then run the test

```
python setup.py test
```

You can also use `pycodestyle`:

```
pycodestyle <script-name>.py
```

### Where should I write my features tests?
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

## Documentation
Write documents of your new `function`/`class`/`feature` and explain what it does.
Writing documents is hard, but it helps others understanding of what you implemented and how to use it.

### Style
We use **numpy style docstring**. <br>
When writing the docs, please follow numpy style. <br>
[See here](https://numpydoc.readthedocs.io/en/latest/) for details. 

## Release
Change the release version in `setup.py` and in `docs/conf.py`
