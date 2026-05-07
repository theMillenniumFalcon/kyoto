from dataclasses import dataclass, field

from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript

from rich.console import Console

console = Console()


@dataclass
class CodeChunk:
    name: str          # function/class name
    kind: str          # "function" | "class" | "method" | "module_top"
    code: str          # actual source code
    language: str
    file_path: str
    repo: str
    start_line: int
    end_line: int
    docstring: str = ""
    imports: list[str] = field(default_factory=list)


# Build language parsers once at import time
_PARSERS: dict[str, Parser] = {
    "python":     Parser(Language(tspython.language())),
    "javascript": Parser(Language(tsjavascript.language())),
    "typescript": Parser(Language(tstypescript.language_typescript())),
}

# Node types that represent chunk boundaries per language
_CHUNK_TYPES: dict[str, set[str]] = {
    "python": {
        "function_definition",
        "class_definition",
        "decorated_definition",
    },
    "javascript": {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "method_definition",
        "export_statement",
    },
    "typescript": {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "method_definition",
        "export_statement",
        "interface_declaration",
        "type_alias_declaration",
    },
}


def _extract_imports(code: str, language: str) -> list[str]:
    imports = []
    for line in code.splitlines()[:30]:
        stripped = line.strip()
        if language == "python" and (
            stripped.startswith("import ") or stripped.startswith("from ")
        ):
            imports.append(stripped)
        elif language in ("javascript", "typescript") and stripped.startswith("import "):
            imports.append(stripped)
    return imports


def _get_docstring(node: Node, source_bytes: bytes) -> str:
    """Extract docstring from the first child string literal (Python only)."""
    for child in node.children:
        if child.type == "block":
            for stmt in child.children:
                if stmt.type == "expression_statement":
                    for inner in stmt.children:
                        if inner.type in ("string", "concatenated_string"):
                            return source_bytes[inner.start_byte:inner.end_byte].decode(
                                errors="ignore"
                            )
    return ""


def _traverse(
    node: Node,
    source_bytes: bytes,
    file_info: dict,
    imports: list[str],
    chunks: list[CodeChunk],
):
    lang = file_info["language"]
    chunk_types = _CHUNK_TYPES.get(lang, set())

    if node.type in chunk_types:
        name_node = node.child_by_field_name("name")
        name = (
            source_bytes[name_node.start_byte:name_node.end_byte].decode()
            if name_node
            else "anonymous"
        )
        code = source_bytes[node.start_byte:node.end_byte].decode(errors="ignore")
        docstring = _get_docstring(node, source_bytes) if lang == "python" else ""

        kind = "function"
        if "class" in node.type or "interface" in node.type:
            kind = "class"
        elif "method" in node.type:
            kind = "method"

        import_header = "\n".join(imports) + "\n\n" if imports else ""

        chunks.append(
            CodeChunk(
                name=name,
                kind=kind,
                code=import_header + code,
                language=lang,
                file_path=file_info["path"],
                repo=file_info["repo"],
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=docstring,
                imports=imports,
            )
        )

    for child in node.children:
        _traverse(child, source_bytes, file_info, imports, chunks)


def parse_file(file_info: dict) -> list[CodeChunk]:
    lang = file_info["language"]
    parser = _PARSERS.get(lang)
    if not parser:
        return []

    try:
        source = file_info["content"]
        source_bytes = source.encode("utf-8")
        tree = parser.parse(source_bytes)
        imports = _extract_imports(source, lang)
        chunks: list[CodeChunk] = []
        _traverse(tree.root_node, source_bytes, file_info, imports, chunks)

        # Fallback: treat whole file as one chunk (small config/utility files)
        if not chunks and len(source.splitlines()) < 200:
            chunks.append(
                CodeChunk(
                    name=file_info["path"].split("/")[-1],
                    kind="module_top",
                    code=source,
                    language=lang,
                    file_path=file_info["path"],
                    repo=file_info["repo"],
                    start_line=1,
                    end_line=len(source.splitlines()),
                    imports=imports,
                )
            )

        return chunks

    except Exception as e:
        console.print(f"[yellow]⚠ Parse error in {file_info['path']}: {e}[/yellow]")
        return []