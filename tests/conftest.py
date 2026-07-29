from pytti.warmup import register_resolvers

# The CLI entry point registers these before composing any config; tests that
# compose configs need them too.
register_resolvers()
