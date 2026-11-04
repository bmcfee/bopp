API
===

Core
----

.. automodule:: bopp.core
    :no-members:
.. currentmodule:: bopp.core
.. autosummary::
    :toctree: generated/
    :nosignatures:

    append
    compute_annotation_id
    create
    create_grouping
    create_metadata
    create_sandbox
    ensure_annotation_id
    extend
    validate
    validate_annotation_id

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

Transforms
----------

.. automodule:: bopp.transforms
    :no-members:
.. currentmodule:: bopp.transforms
.. autosummary::
    :toctree: generated/
    :nosignatures:

    filter_by
    shift
    sort
    to_pixels
    to_score_interval
    to_score_interval_float
    to_score_quarter
    to_score_quarter_float
    to_times
    to_uniform_times
    trim
    union

Evaluation
----------

.. automodule:: bopp.evaluation
    :no-members:
.. currentmodule:: bopp.evaluation
.. autosummary::
    :toctree: generated/
    :nosignatures:

    evaluate

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

Registries
----------

.. automodule:: bopp.registries
    :no-members:
.. currentmodule:: bopp.registries
.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_registry
    get_payload_class
    get_extent_class
    get_confidence_class
    get_metadata_class
    get_grouping_class
    get_annotation_class


Schema-generated objects
------------------------
.. automodule:: bopp.models
.. currentmodule:: bopp.models
.. autosummary::
   :toctree: generated/
   :nosignatures:
   :recursive:

   v1
