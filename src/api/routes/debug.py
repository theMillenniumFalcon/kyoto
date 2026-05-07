from fastapi import APIRouter
from pydantic import BaseModel

from src.debugger.agent import debug_traceback

router = APIRouter()


class DebugRequest(BaseModel):
    traceback: str


@router.post("/debug")
async def debug(req: DebugRequest) -> dict:
    """
    Accept a Python or JavaScript/TypeScript stack trace and return a
    root-cause diagnosis with a suggested fix, powered by Claude tool use.

    Example body:
        {
          "traceback": "Traceback (most recent call last):\\n  File ..."
        }
    """
    return debug_traceback(req.traceback)