.. _api_identify:

API identify
===============

.. automodule:: peerannot.models.identification
   :members:
   :no-inherited-members:

.. currentmodule:: peerannot.models.identification

The identification has three levels of exploration:

* dataset: measure the reliability of the collected votes
* per worker: measure the reliability of the worker
* per task: measure the reliability of the taks

All strategies are available running:

.. prompt:: bash

    peerannot identificationinfo

Dataset exploration
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :recursive:
   :toctree: generated/
   :nosignatures:

   krippendorff_alpha.Krippendorff_Alpha


Per worker exploration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :recursive:
   :toctree: generated/
   :nosignatures:

   Spam_score.Spam_Score
   trace_confusion.Trace_confusion


Per task exploration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :recursive:
   :toctree: generated/
   :nosignatures:

   AUM.AUM
   entropy.Entropy
   WAUM_perworker.WAUM_perworker
   WAUM.WAUM