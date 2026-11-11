import shutil
import subprocess
import tempfile
from pathlib import Path
from secrets import token_urlsafe

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.pool import NullPool


@pytest.fixture(scope="module")
def postgres_owner():
    """Use a new cluster on an owner-only socket, independent of app credentials."""
    yield from _local_postgres("trust")


@pytest.fixture(scope="module")
def scram_postgres_owner():
    yield from _local_postgres("scram-sha-256")


def _local_postgres(authentication):
    initdb, pg_ctl = (shutil.which(name) for name in ("initdb", "pg_ctl"))
    if initdb is None or pg_ctl is None:
        pytest.skip("Local PostgreSQL initdb and pg_ctl are required")

    with tempfile.TemporaryDirectory(prefix="w1-pg-", dir="/tmp") as socket_dir:
        cluster = Path(socket_dir) / "data"
        password = None
        arguments = [initdb, "-D", str(cluster), "-A", authentication, "-U", "w1_owner", "--no-locale", "-E", "UTF8"]
        if authentication == "scram-sha-256":
            password = token_urlsafe(32)
            password_file = Path(socket_dir) / "initdb-password"
            password_file.touch(mode=0o600)
            password_file.write_text(password + "\n")
            arguments.extend(["--pwfile", str(password_file)])
        initialized = subprocess.run(
            arguments,
            capture_output=True, text=True, timeout=30,
        )
        if password is not None:
            password_file.unlink()
        if initialized.returncode and "Operation not permitted" in initialized.stderr:
            pytest.skip("Local PostgreSQL needs shared memory outside this sandbox")
        assert initialized.returncode == 0, initialized.stderr
        subprocess.run(
            [pg_ctl, "-D", str(cluster), "-l", str(Path(socket_dir) / "postgres.log"),
             "-o", f"-c listen_addresses='' -c unix_socket_directories={socket_dir} -c unix_socket_permissions=0700",
             "-w", "start"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        url = URL.create(
            "postgresql+psycopg2", username="w1_owner", password=password,
            database="postgres", query={"host": socket_dir},
        )
        owner = create_engine(url, poolclass=NullPool, hide_parameters=True)
        try:
            yield owner
        finally:
            owner.dispose()
            subprocess.run(
                [pg_ctl, "-D", str(cluster), "-m", "fast", "-w", "stop"],
                check=True, capture_output=True, text=True, timeout=30,
            )
