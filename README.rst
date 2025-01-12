==============
MPI for Python
==============

.. image::  https://github.com/mpi4py/mpi4py/workflows/ci/badge.svg?branch=master
   :target: https://github.com/mpi4py/mpi4py/actions/
.. image::  https://dev.azure.com/mpi4py/mpi4py/_apis/build/status/mpi4py.mpi4py?branchName=master
   :target: https://dev.azure.com/mpi4py/mpi4py/_build
.. image::  https://ci.appveyor.com/api/projects/status/whh5xovp217h0f7n?svg=true
   :target: https://ci.appveyor.com/project/mpi4py/mpi4py
.. image::  https://circleci.com/gh/mpi4py/mpi4py.svg?style=shield
   :target: https://circleci.com/gh/mpi4py/mpi4py
.. image::  https://travis-ci.com/mpi4py/mpi4py.svg?branch=master
   :target: https://travis-ci.com/mpi4py/mpi4py
.. image::  https://scan.coverity.com/projects/mpi4py-mpi4py/badge.svg
   :target: https://scan.coverity.com/projects/mpi4py-mpi4py
.. image::  https://codecov.io/gh/mpi4py/mpi4py/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/mpi4py/mpi4py
.. image::  https://readthedocs.org/projects/mpi4py/badge/?version=latest
   :target: https://mpi4py.readthedocs.org/en/latest/

Overview
--------

Welcome to MPI for Python. This package provides Python bindings for
the *Message Passing Interface* (`MPI <http://www.mpi-forum.org/>`_)
standard. It is implemented on top of the MPI-1/2/3 specification and
exposes an API which grounds on the standard MPI-2 C++ bindings.

Dependencies
------------

* `Python <http://www.python.org/>`_ 2.7, 3.5 or above,
  or `PyPy <http://pypy.org/>`_ 2.0 or above.

* A functional MPI 1.x/2.x/3.x implementation like `MPICH
  <http://www.mpich.org/>`_ or `Open MPI <http://www.open-mpi.org/>`_
  built with shared/dynamic libraries.

* To work with the in-development version, you need to install `Cython
  <http://www.cython.org/>`_.

Testsuite
---------

The testsuite is run periodically on

* `GitHub Actions <https://github.com/mpi4py/mpi4py/actions/>`_

* `Azure Pipelines <https://dev.azure.com/mpi4py/mpi4py>`_

* `AppVeyor <https://ci.appveyor.com/project/mpi4py/mpi4py>`_

* `Circle CI <https://circleci.com/gh/mpi4py/mpi4py>`_

* `Travis CI <https://travis-ci.com/mpi4py/mpi4py>`_

* `Fedora Jenkins <http://jenkins.fedorainfracloud.org/job/mpi4py/>`_
