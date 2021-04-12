Development
===========
TODO: but badges here see paule for reference


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

2. Get a local copy of your fork and install the package in 'development'
   mode, which will make changes in the source code active immediately, by running

   .. code:: bash

       git clone https://github.com/yourname/paule.git
       cd paule
       python setup.py develop --user

3. Add code, tests or documentation.

4. Test your changes locally by running within the root folder (``paule/``)

   .. code:: bash

       make checkstyle
       make test

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

.. note::

    If you want to develop *paule* you should have installed:

    .. code:: bash

        pip install --user tox pylint pytest pycodestyle sphinx


Running tests
-------------

We use ``make`` and ``tox`` to manage testing. You can run the tests by
executing the following within the repository's root folder (``paule/``):

.. code:: bash

    make test

For manually checking coding guidelines run:

.. code:: bash

    make checkstyle

There is an additional way to invoke ``pylint`` as a linter with tox by running

.. code:: bash

    tox -e lint

The linting gives still a lot of complaints that need some decisions on how to
fix them appropriately.


Building documentation
----------------------

Building the documentation requires some extra dependencies. Therefore, run

.. code:: bash

    pip install -e .[docs]

in the project root directory. This command will install all required
dependencies. The projects documentation is stored in the ``paule/doc/`` folder
and is created with ``sphinx``. You can rebuild the documentation by either
executing

.. code:: bash

   make documentation

in the repository's root folder (``paule``) or by executing

.. code:: bash

   make html

in the documentation folder (``paule/doc/``).


Continuous Integration
----------------------

We use several services in order to continuously monitor our project:

===========  ===========  =================  ===========================
Service      Status       Config file        Description
===========  ===========  =================  ===========================
Travis CI    |travis|     `.travis.yml`_     Automated testing
Coveralls    |coveralls|                     Monitoring of test coverage
===========  ===========  =================  ===========================

.. |travis| image:: https://travis-ci.com/quantling/paule.svg?branch=main
    :target: https://travis-ci.com/quantling/paule?branch=main

.. |coveralls| image:: https://coveralls.io/repos/github/quantling/paule/badge.svg?branch=main
    :target: https://coveralls.io/github/quantling/paule?branch=main

.. _.travis.yml: https://github.com/quantling/paule/blob/main/.travis.yml


Licensing
---------

All contributions to this project are licensed under the `GPLv3+ license
<https://github.com/quantling/paule/blob/main/LICENSE.txt>`_. Exceptions are
explicitly marked.
All contributions will be made available under GPLv3+ license if no explicit
request for another license is made and agreed on.


Release Process
---------------
1. Ensure, that the version of the branch to be mered, is adequately increased
   see Versioning_ below.

2. Merge Pull Requests with new features or bugfixes into *paule*'s' ``main``
   branch.

3. Create a new release on Github of the `main` branch of the form ``vX.Y.Z``
   (where ``X``, ``Y``, and ``Z`` refer to the new version).  Add a description
   of the new feature or bugfix. For details on the version number see
   Versioning_ below.

4. Pull the repository and checkout the tag and create the distribution files
   using:

.. code:: bash

    git pull
    git checkout vX.Y.Z
    python setup.py build  # to compile *.pyx -> *.c
    python setup.py sdist

5. Create GPG signatures of the distribution files using:

.. code:: bash

    gpg --detach-sign -a dist/paule-X.Y.Z.tar.gz

6. (maintainers only) Upload the distribution files to PyPI using twine.

.. code:: bash

    twine upload -s dist/*

7. (maintainers only) Check if the new version is on pypi (https://pypi.python.org/pypi/paule/).


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

