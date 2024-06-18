.. _glossary:

Glossary
============

.. list-table::
   :widths: 22 40 40
   :header-rows: 1

   * - Name
     - Definition
     - Mathematical Definition
   * - :math:`n_{task}`
     - The total number of tasks in a dataset
     -
   * - :math:`n_{worker}`
     - The total number of workers in a dataset
     -
   * - :math:`[K]`
     - The set of labels a task can take
     - :math:`[K] = \{1,...,K\}`
   * - :math:`\Delta_K`
     - The simplex of dimension :math:`K-1`, used to represent soft labels (ie. labels as a probability vector along :math:`[K]`)
     - :math:`\Delta_K = \{ p \in [K] : \sum_{k=1}^K p_k=1, p_k \geq 0 \}`
   * - :math:`\mathcal{A(x_i)}`
     - The set of workers that answered the task :math:`i`
     - :math:`\{j\in[ n_{worker} : w_j \text{ answered } x_i\}`
   * - :math:`\mathcal{T(w_j)}`
     - The set of tasks answered by the worker :math:`j`
     - :math:`\{i\in[ n_{task} : w_j \text{ answered }x_i\}`
   * - :math:`\mathcal{Lab(x_i)}`
     - The vector of answered labels of the task :math:`i`
     - :math:`(y_i^{(j)})_{j\in\mathcal{A(x_i)}}`
   * - :math:`y_i^*`
     - The true label of the task :math:`i`
     - :math:`y_i^* \in [K]`
   * - :math:`\hat{y}_i^{agg}`
     - The computed label of the task :math:`i` given the aggregation :math:`agg` method
     - :math:`\begin{cases}\hat{y}_i^{agg} \in [K] \text{ if a hard label} \\ \hat{y}_i^{agg} \in \Delta_K \text{ if a soft label} \end{cases}`
   * - :math:`y^{(j)}_i`
     - The label (hard) that the worker :math:`j` assigned to the task :math:`i`
     -
   * - :math:`\pi^{(j)}`
     - The confusion matrix of the worker :math:`j`
     - :math:`\pi^{(j)}_{k,\ell}=\mathbb{P}(y_i^{(j)​}=\ell∣y_i^\star​=k), \, \forall (\ell,k)\in [K]^2`
   * - :math:`AccTrain(\mathcal{D})`
     - A metric that measure aggregation strategies' accuracies
     - :math:`AccTrain(\mathcal{D}) = \frac{1}{|\mathcal{D}|} \sum_{i=1}^{|\mathcal{D}|} \mathbf{1}_{\Big\{y_i =  \operatorname*{argmax}\limits_{k\in [K]}(ŷ_i)_k\Big\}}`
