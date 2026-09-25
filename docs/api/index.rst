API documentation
=================

Core
----

.. automodule:: bopp.core
    :no-members:
.. currentmodule:: bopp.core
.. autosummary::
    :toctree: generated/
    :nosignatures:

    create
    validate

Input / Output
--------------

.. automodule:: bopp.io
   :no-members:
.. currentmodule:: bopp.io
.. autosummary::
   :toctree: generated/
   :nosignatures:

    load_bopp_json
    save_bopp_json
    load_bopp_msgpack
    save_bopp_msgpack
    load_bopp_csv
    save_bopp_csv

Utilities
---------

.. automodule:: bopp.util
    :no-members:
.. currentmodule:: bopp.util
.. autosummary::
    :toctree: generated/
    :nosignatures:

    to_dataframe
    from_dataframe
    extract_header

Exceptions
----------

.. automodule:: bopp.exceptions
    :no-members:
.. currentmodule:: bopp.exceptions
.. autosummary::
    :toctree: generated/
    :nosignatures:

    BoppError
    BoppValidationError
    BoppArrayError
    BoppRegistryError
    BoppArgumentError
    BoppIOError


Schema-generated objects
------------------------
.. automodule:: bopp.models
.. currentmodule:: bopp.models
.. autosummary::
   :toctree: generated_schema/
   :nosignatures:
   :recursive:

   v1
