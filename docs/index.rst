.. symbolic-dynamics documentation master file, created by
   sphinx-quickstart on Wed May  5 09:20:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

symbolic-dynamics
=================

symbolic-dynamics is a Python package using NetworkX for performing
computations in `symbolic dynamics <http://www.scholarpedia.org/article/Symbolic_dynamics>`_.

>>> import symbolic_dynamics as sd
>>> H = sd.even_shift()
>>> G = sd.mn_gap_shift(4)
>>> sd.is_subshift(G, H)
True
>>> G = sd.mn_gap_shift(3)
>>> sd.find_separating_word(G, H, as_str=True)
'00010001'

Installation
------------
symbolic-dynamics is available on `PyPI <https://pypi.org/>`_. Requires Python 3.8
and above. ::

 > pip install symbolic-dynamics

Reference
---------

.. toctree::
   :maxdepth: 2

   reference
