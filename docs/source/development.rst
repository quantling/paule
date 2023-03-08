Development
===========
TODO: put badges here see pyndl for reference


Getting Involved
----------------
The *paule* project welcomes help in the following ways:

    * Making Pull Requests for
      `code <https://github.com/quantling/paule/tree/main/paule>`_,
      `tests <https://github.com/quantling/paule/tree/main/tests>`_
      or `documentation <https://github.com/quantling/paule/tree/main/doc>`_.
    * Commenting on `open issues <https://github.com/quantling/paule/issues>`_
      and `pull requests <https://github.com/quantling/paule/pulls>`_.
    * Helping to answer `questions in the issue section
      <https://github.com/quantling/paule/labels/question>`_.
    * Creating feature requests or adding bug reports in the `issue section
      <https://github.com/quantling/paule/issues/new>`_.


Workflow
--------

1. Fork this repository on Github. From here on we assume you successfully
   forked this repository to https://github.com/yourname/paule.git

2. Install all dependencies with poetry (https://python-poetry.org/)

   .. code:: bash

       git clone https://github.com/yourname/paule.git
       cd paule
       poetry install

3. Add code, tests or documentation.

4. Test your changes locally by running within the root folder (``paule/``)

   .. code:: bash

        poetry run pytest
        poetry run pylint paule

5. Add and commit your changes after tests run through without complaints.

   .. code:: bash

       git add -u
       git commit -m 'fixes #42 by posing the question in the right way'

   You can reference relevant issues in commit messages (like #42) to make GitHub
   link issues and commits together, and with phrase like "fixes #42" you can
   even close relevant issues automatically.

6. Push your local changes to your fork:

   .. code:: bash

       git push git@github.com:yourname/paule.git

7. Open the Pull Requests page at https://github.com/yourname/paule/pulls and
   click "New pull request" to submit your Pull Request to
   https://github.com/quantling/paule.


Running tests
-------------
We use ``poetry`` to manage testing. You can run the tests by
executing the following within the repository's root folder (``paule/``):

.. code:: bash

    poetry run pytest

For manually checking coding guidelines run:

.. code:: bash

    poetry run pylint paule

The linting gives still a lot of complaints that need some decisions on how to
fix them appropriately.


Building documentation
----------------------
Building the documentation requires some extra dependencies. Usually, these are
installed when installing the dependencies with poetry. Some services like Readthedocs,
however, require the documentation dependencies extra. For that reason, they can
also be found in `docs/requirements.txt`. For normal usage, installing all dependencies
with poetry is sufficient.

The projects documentation is stored in the ``paule/docs/`` folder
and is created with ``sphinx``. However, it is not necessary to build the documentation
from there.

You can rebuild the documentation by either executing

.. code:: bash

    poetry run sphinx-build -b html docs/source docs/build/html

in the repository's root folder (``paule/``) or by executing

.. code:: bash

   poetry run make html

in the documentation folder (``paule/docs/``).


Continuous Integration
----------------------
TODO: see pyndl documentation for reference.


Licensing
---------
All contributions to this project are licensed under the `GPLv3+ license
<https://github.com/quantling/paule/blob/main/LICENSE.txt>`_. Exceptions are
explicitly marked.
All contributions will be made available under GPLv3+ license if no explicit
request for another license is made and agreed on.


Release Process
---------------
1. Update the version accordingly to Versioning_ below. This can be easily done
   by poetry running

   .. code:: bash

       poetry version major|minor|patch|...


2. Merge Pull Requests with new features or bugfixes into *paule*'s' ``main``
   branch.

3. Create a new release on Github of the `main` branch of the form ``vX.Y.Z``
   (where ``X``, ``Y``, and ``Z`` refer to the new version).  Add a description
   of the new feature or bugfix. For details on the version number see
   Versioning_ below. This will trigger a Action to automatically build and
   upload the release to PyPI

4. Check if the new version is on pypi (https://pypi.python.org/pypi/paule/).

5. Manuel publishing works the following (maintainer only):

   .. code:: bash

      git pull
      git checkout vX.Y.Z
      poetry build
      poetry publish


Versioning
----------
We use a semvers versioning scheme. Assuming the current version is ``X.Y.Z``
than ``X`` refers to the major version, ``Y`` refers to the minor version and
``Z`` refers to a bugfix version.

Bugfix release
^^^^^^^^^^^^^^
For a bugfix only merge, which does not add any new features and does not
break any existing API increase the bugfix version by one (``X.Y.Z ->
X.Y.Z+1``).

Minor release
^^^^^^^^^^^^^
If a merge adds new features or breaks with the existing API a deprecation
warning has to be supplied which should keep the existing API. The minor
version is increased by one (``X.Y.Z -> X.Y+1.Z``). Deprecation warnings should
be kept until the next major version. They should warn the user that the old
API is only usable in this major version and will not be available any more
with the next major ``X+1.0.0`` release onwards. The deprecation warning should
give the exact version number when the API becomes unavailable and the way of
achieving the same behaviour.

Major release
^^^^^^^^^^^^^
If enough changes are accumulated to justify a new major release, create a new
pull request which only contains the following two changes:

- the change of the version number from ``X.Y.Z`` to ``X+1.0.0``
- remove all the API with deprecation warning introduced in the current
  ``X.Y.Z`` release

