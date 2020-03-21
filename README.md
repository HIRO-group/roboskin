# How to add documentations
## Steps
Run all the commands in the package home directory!!! 
1. Write docstring comments in your source
2. Run `sphinx-apidoc -f -o docs robotic_skin` to generate `rst` files from the python module
3. Run `make html` (in `docs` directory) or `sphinx-autobuild docs docs/_build/html` to generate `html` files
4. Access [http://127.0.0.1:8000 ](http://127.0.0.1:8000) locally.
5. If everyting looks good, push commits. GitHub Page will look into `index.html` first. This file will then look into `html` files in `docs/_build/html`. GitHub Page will autogenerates the documentations from those files.

## Structure
`index.rst` is the main file. It refers to `robotic_skin.rst` file. <br>

```
└── index.rst 
    └── robotic_skin.rst
```

# Run Locally before submitting
In the package home directory, run
```
sphinx-autobuild docs docs/_build/html
``` 
Access [http://127.0.0.1:8000 ](http://127.0.0.1:8000) to check if everything looks good.


# Why can't I see `docs/_build` directory?
You will not see `docs/_build` directory online. It is added to `.gitignore`.


# How should I start building documentations from scratch???
[Follow Sphinx's Official Tutorial](https://www.sphinx-doc.org/en/master/usage/quickstart.html)

1. Run `sphinx-apidoc -f -o docs robotic_skin` in the package home directory
2. Edit `conf.py` as in our example.
3. Add `index.html` as in our example.
4. Now you have exactly the same structure as ours. Follow the steps above.

# [How to set a GitHub Page using Sphinx](https://github.com/sphinx-doc/sphinx/issues/3382) 
This GitHub thread answers how to release our page using GitHub Page. 

# Sphinx Commands
There were few commands that confused me so I will list them with their functions. More information can be found [here](https://www.sphinx-doc.org/en/master/man/index.html). You can also generate `pdf` instead of `html` files, but we won't do it anyways. 

- `sphinx-quickstart` <br>
Generates essential files like `Makefile`, `conf.py` and `index.rst`<br>

- `sphinx-build`<br>
Generates documentation from `rst` files in source directory and output `html` files to output directory.<br>

- `sphinx-apidoc` <br>
Auto-Generates a documentation from Python Package.
`sphinx-apidoc _build/html ../robotic_skin` will generates `rst` files automatically.<br>
Usage: `sphinx-apidoc <OUTPUT_DIR> <MODULE_DIR>`

- `sphinx-autobuild` (after pip installed it)<br>
Generates `html` files and run server to host it to see it locally.<br>
Usage: `sphinx-autobuild <SOURCE_DIR> <OUTPUT_DIR>`
