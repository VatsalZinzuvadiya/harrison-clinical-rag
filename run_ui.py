"""
Streamlit Launcher Entrypoint

Applies Starlette 0.45+ runtime compatibility patch prior to Streamlit CLI initialization.
Allows running Streamlit without any version mismatch errors between FastAPI and Streamlit.
"""

import starlette.middleware.gzip as _gz

if not hasattr(_gz, "DEFAULT_EXCLUDED_CONTENT_TYPES"):
    setattr(_gz, "DEFAULT_EXCLUDED_CONTENT_TYPES", ("text/html", "text/css", "text/javascript", "text/plain", "application/json"))
if not hasattr(_gz, "IdentityResponder"):
    setattr(_gz, "IdentityResponder", type("IdentityResponder", (), {}))

import sys
from streamlit.web.cli import main

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "ui.py"]
    sys.exit(main())
