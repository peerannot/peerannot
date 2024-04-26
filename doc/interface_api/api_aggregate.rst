.. _api_aggregate:

API aggregate
===============

.. automodule:: peerannot.models.aggregation
   :members:
   :no-inherited-members:

.. currentmodule:: peerannot.models.aggregation

The ``aggregate`` strategies are used to combine labels from multiple users into a single label.
This aggregated label can be either a probability distribution (soft label) or a single class label (hard label).

All strategies are located at `this direction on GitHub <https://github.com/peerannot/peerannot/tree/main/peerannot/models/aggregation>`.

All aggregation-based strategies are available running:

.. prompt:: bash

    peerannot agginfo

.. autosummary::
   :recursive:
   :toctree: generated/
   :nosignatures:

    MV.MV
    NaiveSoft.NaiveSoft
    Wawa.Wawa
    IWMV.IWMV
    DS.Dawid_Skene
    GLAD.GLAD
    plantnet.PlantNet
    twothird.TwoThird
    DS_clust.Dawid_Skene_clust
    WDS.WDS
