.. _cli_simulate:

CLI simulate
===============

The help documentation is available in the terminal from:

.. prompt:: bash

    peerannot simulate --help


Simulate independent mistakes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The independent mistakes setting considers that each worker :math:`w_j` answers following a multinomial distribution with weights given at the row :math:`y_i^{\star}` of their confusion matrix :math:`\pi^{(j)}`. Each confusion row in the confusion matrix is generated uniformly in the simplex. Then, we make the matrix diagonally dominant (to represent non-adversarial workers) by switching the diagonal term with the maximum value by row. Answers are independent of one another as each matrix is generated independently and each worker answers independently of other workers. In this setting, the DS model is expected to perform better with enough data as we are simulating data from its assumed noise model.

.. prompt:: bash

    peerannot simulate --n-worker=30 --n-task=200  --n-classes=5 \
                     --strategy independent-confusion \
                     --feedback=10 --seed 0 \
                     --folder ./simus/independent

This example generates 200 tasks and 30 workers with :math:`K=5` classes. Each task receivex :math:`\mathcal{A}(x_i)=10` votes. This leads to around :math:`200\times 10/30\simeq 67` tasks per worker (variations are due to the randomness in the affectations).

.. note::

    To create imbalanced number of votes per task, use the ``--imbalance-votes`` option. The number of votes is then chosen at random uniformly between 1 and the number of workers available.

Simulate correlated mistakes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The correlated mistakes are also known as the student-teacher or junior-expert setting (Cao et al. (2019)). Consider that the crowd of workers is divided into two categories: teachers and students (with :math:`n_{teacher}+n_{student}=n_{worker}`). Each student is randomly assigned to one teacher at the beginning of the experiment. We generate the (diagonally dominant) confusion matrices of each teacher and the students share the same confusion matrix as their associated teacher. Hence, clustering strategies are expected to perform best in this context. Then, they all answer independently, following a multinomial distribution with weights given at the row :math:`y_i^{\star}` of their confusion matrix :math:`\pi^{(j)}`.


.. prompt:: bash

    peerannot simulate --n-worker=30 --n-task=200  --n-classes=5 \
                     --strategy student-teacher \
                     --ratio 0.8 \
                     --feedback=10 --seed 0 \
                     --folder ./simus/student_teacher

This example generates 200 tasks and 30 workers with :math:`K=5` classes. Each task receivex :math:`\mathcal{A}(x_i)=10` votes. There are 80% of students in the crowd, defined by the ``--ratio`` parameter.

Simulate mistakes with discrete difficulty levels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Introduced in Whitehill et al. (2009), workers are either good or bad. Tasks are either easy or hard. The keyword ``ratio-diff`` indicates the prevalence of each level of difficulty as the ratio of easy tasks over hard tasks:

.. math::

    \texttt{ratio-diff} = \frac{\mathbb{P}(\texttt{easy})}{\mathbb{P}(\texttt{hard})}, \mathbb{P}(\texttt{easy})+\mathbb{P}(\texttt{hard})=1

.. prompt:: bash

    peerannot simulate --n-worker=100 --n-task=200  --n-classes=5 \
                     --strategy discrete-difficulty \
                     --ratio 0.35 --ratio-diff 1 \
                     --feedback 10 --seed 0 \
                     --folder ./simus/discrete_difficulty

We simulate 200 tasks and 100 workers with :math:`K=5` classes. Each task receives :math:`\mathcal{A}(x_i)=10` votes. The ratio of good workers is 0.35. The ratio of easy tasks is 1. 35% of workers are good and there is 50% of easy tasks.

.. click:: peerannot.runners.simulate:simulate
    :prog: peerannot simulate
    :nested: full
    :commands: simulate