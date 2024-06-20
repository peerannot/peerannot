.. _add_strategy:

Add a strategy in peerannot
=====================================

This tutorial shows how to add a new strategy in ``peerannot``.


.. Hint::
    If not yet done, please go to :ref:`get started page <get_started>` to install the ``peerannot`` library.


What is an aggregation?
-------------------------

The goal is, from a set of collected labels, to obtain a single label (or label distribution) for each task following an aggregation :math:`\texttt{agg}`.

.. math::

    \forall i\in[n_\texttt{task}],\texttt{agg}(\{y_i^{(j)}\}_{j\in\mathcal{A}(x_i)}) = \hat y_i\in\Delta_K

If the aggregation results in a distribution that is not a Dirac, the label is called a `soft` label. Otherwise it is a `hard` label.

.. Hint::
    All aggregations in ``peerannot`` are in the folder at ``peerannot/models/aggregation`` and should be added in the ``__init__.py`` file at ``peerannot.models/`` once validated to be used through the CLI.

Iterative aggregations
--------------------------

An iterative aggregation strategy is like Dawid and Skene's algorithm.
There is an initialization and then an optimization-like process to update the label distribution.

.. code-block:: python

    from ..template import CrowdModel

    class MyAggregation(CrowdModel):
        def __init__(self, answers, n_classes, n_workers, **kwargs):
            ... # instantiate all your parameters

        def run(self, maxiter=10, **kwargs):
            # Initialization
            self.init()

            # Iterative process
            for i in range(max_iter):
                ... # update the label distribution
                ... # check convergence

        def get_probas(self):
            ... # return label distributions

        def get_answers(self):
            ... # return hard labels

If the aggregation is not able to return label distributions per task, raise a warning and redirect to the ``ŋet_answers`` method.
For example:

.. code-block:: python

    def get_probas(self):
        warnings.warn(
            """
            MyAggregation only returns hard labels.
            Defaulting to ``get_answers()``.
            """
        )
        return self.get_answers()

Not iterative aggregations
----------------------------

For non iterative aggregations (like Majority vote), simply drop the ``run`` method and implement the ``get_probas`` and ``get_answers`` methods.