# "ab" is shared by NN-Lite and the NN Dataset. Extending the search path lets
# ab.lite from an nn-lite checkout and ab.nn from the installed nn-dataset
# package be imported together.
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
