import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

ext_modules = [
    Extension("main",  ["main.py"]),
    Extension("custom_logger",  ["custom_logger.py"]),
    Extension("config",  ["config.py"]),
    Extension("animal",  ["env/animal.py"]),
    Extension("fish",  ["env/fish.py"]),
    Extension("shark",  ["env/shark.py"]),
    Extension("aquarium",  ["env/aquarium.py"]),
    Extension("util",  ["env/util.py"]),
    Extension("animal_controller",  ["env/animal_controller.py"]),
    Extension("collision",  ["env/collision.py"])
    # Extension("view",  ["view.py"])
]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"}

setup(
    name='aquarium_experiments',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
