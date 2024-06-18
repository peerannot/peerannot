``peerannot``
=============

*--Handling your crowdsourced datasets easily--*


|Python 3.8+| |PyPI version|

The peerannot library was created to handle crowdsourced labels in classification problems.


Getting started
===============

Start here to get up and running

.. toctree::
   :maxdepth: 2

   get_started

Tutorials and additional examples
=====================================

Want to dive deeper into the library? Check out the tutorials
You will find ressources to add your own datasets, strategy, and run your first label aggregations.

.. toctree::
    :maxdepth: 2

    tutorials/index

* More examples can be found in the `published paper in Computo Journal <https://computo.sfds.asso.fr/published-202402-lefort-peerannot/>`_


API and CLI Reference
=====================

Want to deep dive into the library?
In addition to the tutorials, you can find the full API and CLI reference here.

.. grid:: 2

   .. grid-item-card:: **API Reference**
      :link: interface_api
      :link-type: ref

      :octicon:`book` **Run peerannot from a python script**
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   .. grid-item-card:: **CLI Reference**
      :link: interface_cli
      :link-type: ref

      :octicon:`terminal` **Run peerannot from your terminal**
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :hidden:
   :includehidden:

   interface_api/index
   interface_cli/index


.. include:: glossary.rst


.. toctree::
   :maxdepth: 1

   glossary


Citation
============

Cite us, join us, and let us collaboratively improve our toolbox!

.. code-block:: bibtex

   @article{lefort2024,
      author = {Lefort, Tanguy and Charlier, Benjamin and Joly, Alexis and Salmon, Joseph},
      publisher = {French Statistical Society},
      title = {Peerannot: Classification for Crowdsourced Image Datasets with {Python}},
      journal = {Computo},
      date = {2024-04-04},
      url = {https://computo.sfds.asso.fr/published-202402-lefort-peerannot/},
   }


.. |Python 3.8+| image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
.. |PyPI version| image:: https://badge.fury.io/py/peerannot.svg
   :target: https://pypi.org/project/peerannot/
