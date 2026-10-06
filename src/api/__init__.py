"""FastAPI application.

The app factory is create_app(); `app` is the module-level instance uvicorn
serves. Both live in src.api.app.
"""

from src.api.app import create_app

__all__ = ["create_app"]
