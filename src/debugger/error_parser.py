import re
from dataclasses import dataclass, field


@dataclass
class StackFrame:
    file: str
    line: int
    function: str
    snippet: str = ""


@dataclass
class ParsedError:
    language: str
    error_type: str
    message: str
    frames: list[StackFrame] = field(default_factory=list)


def parse_python(tb: str) -> ParsedError:
    lines = tb.strip().splitlines()
    error_line = lines[-1]
    parts = error_line.split(":", 1)
    etype = parts[0].strip()
    emsg = parts[1].strip() if len(parts) > 1 else ""

    frames = []
    pattern = re.compile(r'File "(.+?)", line (\d+), in (.+)')
    for i, line in enumerate(lines):
        m = pattern.search(line)
        if m:
            snippet = lines[i + 1].strip() if i + 1 < len(lines) else ""
            frames.append(
                StackFrame(
                    file=m.group(1),
                    line=int(m.group(2)),
                    function=m.group(3),
                    snippet=snippet,
                )
            )

    return ParsedError(language="python", error_type=etype, message=emsg, frames=frames)


def parse_javascript(tb: str) -> ParsedError:
    lines = tb.strip().splitlines()
    error_line = lines[0]
    parts = error_line.split(":", 1)
    etype = parts[0].strip()
    emsg = parts[1].strip() if len(parts) > 1 else ""

    frames = []
    # "    at functionName (file.js:42:10)"
    # "    at file.js:42:10"  (anonymous)
    named = re.compile(r"at (.+?) \((.+?):(\d+):\d+\)")
    anon = re.compile(r"at (.+?):(\d+):\d+$")
    for line in lines[1:]:
        m = named.search(line)
        if m:
            frames.append(
                StackFrame(file=m.group(2), line=int(m.group(3)), function=m.group(1))
            )
            continue
        m = anon.search(line.strip())
        if m:
            frames.append(
                StackFrame(file=m.group(1), line=int(m.group(2)), function="<anonymous>")
            )

    return ParsedError(language="javascript", error_type=etype, message=emsg, frames=frames)


def _strip_env_prefix(path: str) -> str:
    """Strip common env prefixes (/app/, /home/user/project/) so paths match
    the relative paths stored in Pinecone during ingestion."""
    # Remove leading absolute path segments up to the first recognisable src dir
    for prefix in ("/app/", "/usr/src/app/", "/workspace/", "/project/"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def parse_traceback(tb: str) -> ParsedError:
    """Auto-detect language and parse the traceback."""
    if "Traceback (most recent call last)" in tb:
        error = parse_python(tb)
    elif " at " in tb and ("Error:" in tb or "Exception:" in tb):
        error = parse_javascript(tb)
    else:
        return ParsedError(language="unknown", error_type="UnknownError", message=tb[:200])

    # Normalise file paths so they match indexed relative paths
    for frame in error.frames:
        frame.file = _strip_env_prefix(frame.file)

    return error