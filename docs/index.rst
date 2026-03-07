Pendulum Sim Documentation
==========================

This project models a double-pendulum suspension (LIGO-inspired) and compares:

* passive behavior (no control force), and
* active control using RL (PPO) on the top mass.


Why graphs might not appear on ReadTheDocs
------------------------------------------

ReadTheDocs builds static documentation from files in ``docs/``.
It does **not** run long RL training jobs or pop up Matplotlib windows.
So plots only appear on RTD if you:

1. generate image files locally (``.png``), and
2. commit those files, and
3. reference them from this page.

If you only run ``python pend_rl.py`` locally without committing generated images,
RTD will show text only.


How to generate plots locally
-----------------------------

From the repository root:

.. code-block:: bash

   python pend_rl.py

This saves files like:

* ``rl_result_seedXXXXX.png`` (time-domain displacement + force)
* ``rl_asd_seedXXXXX.png`` (ASD comparison)
* ``rl_learning_curve.png`` (training reward trend)


How to include plots in docs
----------------------------

After generating plots, copy/rename them into a tracked docs folder (for example
``docs/_static/``), commit them, and reference them:

.. code-block:: rst

   .. image:: _static/rl_result_example.png
      :alt: RL vs passive displacement and force
      :width: 900px


What “good” RL behavior should look like
-----------------------------------------

A healthy controller run should typically show:

* **Time domain:** RL displacement (blue) with smaller amplitude than passive (gray).
* **ASD panel:** RL displacement ASD below passive over key low-frequency bands.
* **Force panel:** control force is dynamic (not a flat near-zero line and not always saturated).

If RL and passive overlap heavily and force stays near zero, the policy likely found a
weak local minimum (under-actuation / doing nothing).


Common Git conflict recovery (your checkout issue)
--------------------------------------------------

If ``git pull origin main`` reports merge conflicts and blocks checkout:

.. code-block:: bash

   # 1) See conflicted files
   git status

   # 2) Open and resolve conflict markers in file(s), e.g. pend_rl.py
   #    <<<<<<<, =======, >>>>>>>

   # 3) Mark as resolved
   git add pend_rl.py

   # 4) Finish merge commit
   git commit -m "Resolve merge conflict in pend_rl.py"

   # 5) Now checkout/pull/push works again
   git checkout main
   git pull origin main
   git push origin main

If you want to abandon the in-progress merge instead:

.. code-block:: bash

   git merge --abort

Then retry your pull strategy cleanly.
