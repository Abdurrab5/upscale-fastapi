import os

os.environ.setdefault("CPANEL_ENV", "1")
os.environ.setdefault("SKIP_WARMUP", "1")

from app import app

# Passenger-compatible entrypoint for cPanel/shared hosting.
application = app
