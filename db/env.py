import os
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from sqlmodel import SQLModel
from dotenv import load_dotenv

# Load .env file for DATABASE_URL
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override sqlalchemy.url with env var
config.set_main_option("sqlalchemy.url", DB_URL)

# Import models and set target metadata
from app.db_models import JobTitle

target_metadata = SQLModel.metadata


def include_object(object, name, type_, reflected, compare_to):
    # Skip index comparison for HNSW (custom index Postgres-specific)
    if type_ == "index" and "hnsw" in name:
        return False
    return True


def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    context.configure(
        url=DB_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=False,
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=False,
            include_object=include_object,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
