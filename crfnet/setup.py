import setuptools
from setuptools.extension import Extension
from distutils.command.build_ext import build_ext as DistUtilsBuildExt

class BuildExtension(setuptools.Command):
    description     = DistUtilsBuildExt.description
    user_options    = DistUtilsBuildExt.user_options
    boolean_options = DistUtilsBuildExt.boolean_options
    help_options    = DistUtilsBuildExt.help_options

    def __init__(self, *args, **kwargs):
        from setuptools.command.build_ext import build_ext as SetupToolsBuildExt

        # Bypass __setatrr__ to avoid infinite recursion.
        self.__dict__['_command'] = SetupToolsBuildExt(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._command, name)

    def __setattr__(self, name, value):
        setattr(self._command, name, value)

    def initialize_options(self, *args, **kwargs):
        return self._command.initialize_options(*args, **kwargs)

    def finalize_options(self, *args, **kwargs):
        ret = self._command.finalize_options(*args, **kwargs)
        import numpy
        self.include_dirs.append(numpy.get_include())
        return ret

    def run(self, *args, **kwargs):
        return self._command.run(*args, **kwargs)


extensions = [
    Extension(
        'utils.compute_overlap',
        ['utils/compute_overlap.pyx'],
    ),
]

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name             = 'crfnet',
    version          = '0.99.0',
    description      = 'Installing all required packages for CRF-Net',
    url              = '',
    author           = 'Markus Weber',
    author_email     = 'markus.weber@tum.de',
    maintainer       = 'Markus Weber',
    maintainer_email = 'markus.weber@tum.de',
    install_requires = requirements,
    # packages         = ["crfnet"], #setuptools.find_packages(where='.', exclude=(), include=('*',)),
    # package_dir = {'': 'crfnet'},
    ext_modules    = extensions,
    cmdclass         = {'build_ext': BuildExtension},
    setup_requires = ["cython>=0.28", "numpy>=1.16.0"]
)

