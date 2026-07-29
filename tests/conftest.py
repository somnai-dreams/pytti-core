# Registered by the CLI entry point before composing any config; tests that
# compose configs need the same setup.
import pytti.config.structured_config  # noqa: F401  (registers the schema)
from pytti.warmup import register_resolvers

register_resolvers()
