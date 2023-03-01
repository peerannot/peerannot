Peerannot: learning from crowdsourced labels
==============================================

Crowdsourcing labels is now a standard practice, but how do we learn from them?
Peerannot allow users to easily manipulate different aggregation or end-to-end (E2E) strategies in order to easily produce results.

.. note::

   This library is under active development and breakable changes might happen until the first major release.
   TODO: add some colors and links to references

.. toctree::
    :maxdepth: 1

    strategies

Crowdsourcing researchers often follow one of two paths:

* Aggregate the labels and then learn from the resulting aggregation,
* End-to-end (E2E) strategies that directly introduces all the labels into an adapted neural network.

TODO: add the scheme of label aggregation from the paper
The aggregation strategies can be accessed using the keyword `aggregate`. in the CLI

TODO: make an example

Among the one available, there is the majority vote, Dawid and Skene model or GLAD (non exhaustive list).
All strategies available for label aggregation are displayed using

.. prompt:: bash $

   peerannot agginfo

Once labels are aggregated, the `run` method can produce easily reproducible results for you to explore.
By default, we monitor the following metrics (the loss used is the crossentropy loss):

* train loss
* validation loss (if validation set is available)
* validation accuracy (if validation set is available)
* test loss
* test accuracy
* test expected calibration error (ECE)
* test expected calibration error by class (Macro ECE)

Use the following command to know more on how to train standard neural networks with peerannot:

.. prompt:: bash $

   peerannot train --help


.. toctree::
    :maxdepth: 1

    datasets


