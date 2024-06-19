.. _api_aggregate_deep:

API aggregate_deep
=====================

.. automodule:: peerannot.models.agg_deep
   :members:
   :no-inherited-members:

.. currentmodule:: peerannot.models.agg_deep

The ``aggregate-deep`` strategies are end-to-end strategies that include an aggregation-like layer in the architecture of the computer vision model.

All computer vision models that can be used for training are available through the ``Torchvision`` library and can be found running:

.. prompt:: bash

    peerannot modelinfo

All aggregation-based strategies are available running:

.. prompt:: bash

    peerannot agginfo

All strategies are located at `this direction on GitHub <https://github.com/peerannot/peerannot/tree/main/peerannot/models/agg_deep>`

.. autosummary::
   :recursive:
   :toctree: generated/
   :nosignatures:

   CoNAL.CoNAL
   Crowdlayer.Crowdlayer

For more specifications on the architectures and blocks of the networks, please visit `the strategy file documentation on GitHub <https://github.com/peerannot/peerannot/tree/main/peerannot/models/agg_deep>`.
