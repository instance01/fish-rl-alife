import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

ext_modules = [
    Extension("pipeline",  ["pipeline.pyx"]),
    Extension("custom_logger",  ["custom_logger.pyx"]),
    Extension("config",  ["config.pyx"]),
    # Extension("animal",  ["env/animal.py"]),  # Doesn't work due to ABC?
    Extension("env.fish",  ["env/fish.pyx"]),
    Extension("env.shark",  ["env/shark.pyx"]),
    Extension("env.aquarium",  ["env/aquarium.pyx"]),
    Extension("env.util",  ["env/util.pyx"]),
    Extension("env.animal_controller",  ["env/animal_controller.pyx"]),
    Extension("env.collision",  ["env/collision.pyx"])
    # Extension("view",  ["view.py"])
]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"}

setup(
    name='aquarium_experiments',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
