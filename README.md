Make sure Python is installed on the system by checking its version. Otherwise, download it [here](https://www.python.org/downloads/).

```
python --version
```

Navigate to the root directory of the repository. Then, install the required dependencies using the package installer for Python.

```
pip install -r requirements.txt
```

For automatic name-field OCR in the UI, install the Tesseract OCR runtime as well. On Windows, the UI will auto-detect a standard Tesseract install, or you can point it to a custom install with the `TESSERACT_CMD` environment variable.

Example forms saved in `test_forms/` are omitted due to privacy reasons and are available upon request from the repository owner.

`test_grader.py` can be used if the directory `test_forms/` exists with `.png` files. 

An example command with terminal navigated to the repository root:

```
python test_grader.py -i test_forms\a.png
```

The UI can be run by executing the command `run` in the root directory (recommended).