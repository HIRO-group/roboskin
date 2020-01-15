# How to add documentations
## Steps
Run all the commands in `docsrc` directory!!! 
1. (If not done) Add `robotic_skin` in `index.rst file`
2. Run `sphinx-apidoc -f -o . ../robotic_skin/` to generate `rst` files from the python module
3. Run `sphinx-autobuild . ../docs` to generate `html` files under `docs` directory
4. Access [http://127.0.0.1:8000 ](http://127.0.0.1:8000) locally.
5. If everyting looks good, push commits. GitHub Page will automatically finds the `html` files in `docs` directory and you are done!

## Structure
`index.rst` is the main file. It refers to other `.rst` files. <br>

```
└── index.rst 
    └── robotic_skin.rst
```

# [Why do we have both docsrc and docs dicrectory?](https://github.com/sphinx-doc/sphinx/issues/3382) 
This GitHub thread answers why we have `docs` and `docsrc`. 
GitHub Pages only allows users to locate `html` files in `docs`. I thought it would be very messy to have both `rst` and `html` files in one directory so I separated. <br>

## `docsrc`
This is where `rst` files are. Run `sphinx-apidoc -f -o . ../robotic_skin/` in this directory. The command will create `rst` files from python module `robotic_skin`

## `docs` 
This is where generated `html` files are. 
GitHub Page will look for `html` files in this directory.

# Run Locally before submitting
The command will generate `html` files from `rst` files and run a host server locally. <br>
Access [http://127.0.0.1:8000 ](http://127.0.0.1:8000) to check if everything looks good.

Under `docsrc` directory, run
```
sphinx-autobuild . ../docs 
```

# Why can't I see `docsrc/_build` directory?
You will not see `docsrc/_build` directory online. It is added to `.gitignore`.


# How should I start building documentations from scratch???
[Follow Sphinx's Official Tutorial](https://www.sphinx-doc.org/en/master/usage/quickstart.html)

1. Run `sphinx-quickstart` under `docsrc` directory.
2. Answer all the questions and it will auto-generate files
3. Edit `conf.py` as in our example.
4. Now you have exactly the same structure as ours. Follow the steps above.

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

