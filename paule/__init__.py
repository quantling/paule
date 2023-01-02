"""
paule - Predictive Articulatory speech synthesis Utilizing Lexical Embeddings
=============================================================================

*paule* is a prediction based control and copy synthesis model for the
articulatory speech synthesiser VocalTractLab (VTL; vocaltractlab.de). *paule* can
plan control parameter (cp) trajectories for the VTL synthesiser for a target
acoustics or target lexical or semantic embedding. To achieve this it uses
an inverse and and a predictive sequence-to-sequence model and a
sequence-to-vector classifier model. All models are
implemented in pytorch and pretrained weights are available.

"""

print("WARNING! The *paule* package is still in alpha. "
      "To download pretrained weights for the PAULE model, which is defined in "
      "`paule.paule.Paule`, you can invoke "
      "`paule.util.download_pretrained_weights()`.")

import os
import sys
import multiprocessing as mp
from pip._vendor import pkg_resources
try:
    from importlib.metadata import requires
except ModuleNotFoundError:  # python 3.7 and before
    requires = None
try:
    from packaging.requirements import Requirement
except ModuleNotFoundError:  # this should only happend during setup phase
    Requirement = None


__author__ = 'Konstantin Sering, Paul Schmidt-Barbo'
__author_email__ = 'konstantin.sering@uni-tuebingen.de'
__version__ = '0.3.0'
__license__ = 'GPLv3+'
__description__ = ('paule - Predictive Articulatory speech synthesis Utilizing Lexical Embeddings')
__classifiers__ = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License (GPL)',
    'Operating System :: POSIX :: Linux',
    'Operating System :: MacOS',
    'Operating System :: Microsoft :: Windows',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Information Analysis',
    ]


def sysinfo():
    """
    Prints system the dependency information
    """
    if requires:
        dependencies = [Requirement(req).name for req in requires('paule')
                        if not Requirement(req).marker]

    header = ("Paule Information\n"
              "=================\n\n")

    general = ("General Information\n"
               "-------------------\n"
               "Python version: {}\n"
               "Paule version: {}\n\n").format(sys.version.split()[0], __version__)

    uname = platform.uname()
    osinfo = ("Operating System\n"
              "----------------\n"
              "OS: {s.system} {s.machine}\n"
              "Kernel: {s.release}\n"
              "CPU: {cpu_count}\n").format(s=uname, cpu_count=mp.cpu_count())

    if uname.system == "Linux":
        _, *lines = os.popen("free -m").readlines()
        for identifier in ("Mem:", "Swap:"):
            memory = [line for line in lines if identifier in line]
            if len(memory) > 0:
                _, total, used, *_ = memory[0].split()
            else:
                total, used = '?', '?'
            osinfo += "{} {}MiB/{}MiB\n".format(identifier, used, total)

    osinfo += "\n"

    deps = ("Dependencies\n"
            "------------\n")

    if requires:
        deps += "\n".join("{pkg.__name__}: {pkg.__version__}".format(pkg=__import__(dep))
                          for dep in dependencies)
    else:
        deps = 'You need Python 3.8 or higher to show dependencies.'

    print(header + general + osinfo + deps)

