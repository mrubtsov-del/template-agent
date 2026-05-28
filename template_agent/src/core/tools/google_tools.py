"""Google Workspace tools for the agent.

Provides LangChain tools to read data from Google Sheets and Google Docs
when a user shares a link in chat. Uses the service account configured
via GOOGLE_APPLICATION_CREDENTIALS_CONTENT.
"""

from __future__ import annotations

import os
import re
from typing import Any

from langchain_core.tools import tool

from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)

_SHEETS_RE = re.compile(r"docs\.google\.com/spreadsheets/d/([a-zA-Z0-9_-]+)")
_DOCS_RE = re.compile(r"docs\.google\.com/document/d/([a-zA-Z0-9_-]+)")
_BARE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{20,}$")

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
]


def _extract_id(url_or_id: str, pattern: re.Pattern[str]) -> str | None:
    m = pattern.search(url_or_id)
    if m:
        return m.group(1)
    if _BARE_ID_RE.fullmatch(url_or_id.strip()):
        return url_or_id.strip()
    return None


def _build_credentials():
    """Return google.oauth2 service-account Credentials from the configured file."""
    from google.oauth2 import service_account

    creds_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_file or not os.path.exists(creds_file):
        raise RuntimeError(
            "Google service account credentials not available. "
            "Set GOOGLE_APPLICATION_CREDENTIALS_CONTENT in the environment."
        )
    return service_account.Credentials.from_service_account_file(
        creds_file, scopes=_SCOPES
    )


def _doc_body_to_text(body: dict[str, Any]) -> str:
    """Recursively extract plain text from a Google Docs document body."""
    lines: list[str] = []
    for element in body.get("content", []):
        paragraph = element.get("paragraph")
        table = element.get("table")
        if paragraph:
            text_parts: list[str] = []
            for pe in paragraph.get("elements", []):
                tr = pe.get("textRun")
                if tr:
                    text_parts.append(tr.get("content", ""))
            line = "".join(text_parts).rstrip("\n")
            if line:
                lines.append(line)
        elif table:
            for row in table.get("tableRows", []):
                cells: list[str] = []
                for cell in row.get("tableCells", []):
                    cell_text = _doc_body_to_text(cell.get("content", {}) if isinstance(cell.get("content"), dict) else {"content": cell.get("content", [])})
                    cells.append(cell_text.strip())
                lines.append(" | ".join(cells))
    return "\n".join(lines)


@tool
def read_google_sheet(url: str, sheet_name: str | None = None) -> dict[str, Any]:
    """Read data from a Google Sheet shared via URL or spreadsheet ID.

    Call this whenever the user provides a Google Sheets link or ID. Reads up
    to 10 000 cells. Returns column headers and rows so you can summarise or
    analyse the data immediately.

    Args:
        url: Google Sheets URL (https://docs.google.com/spreadsheets/d/...) or
            bare spreadsheet ID.
        sheet_name: Tab name to read. Reads the first visible sheet when omitted.

    Returns:
        Dict with keys ``spreadsheet_title``, ``sheet_name``, ``columns``,
        ``rows`` (list-of-lists), ``row_count``, and ``truncated``.
        On error: ``{"error": ..., "error_type": ...}``.
    """
    spreadsheet_id = _extract_id(url, _SHEETS_RE)
    if not spreadsheet_id:
        return {
            "error": f"Could not parse a Google Sheets ID from: {url!r}",
            "error_type": "invalid_url",
        }

    logger.info("google_sheets.read.start spreadsheet_id=%s sheet=%s", spreadsheet_id, sheet_name)
    try:
        from googleapiclient.discovery import build  # type: ignore[import-untyped]

        creds = _build_credentials()
        service = build("sheets", "v4", credentials=creds, cache_discovery=False)
        spreadsheets = service.spreadsheets()

        # Resolve metadata to get sheet names and title
        meta = spreadsheets.get(spreadsheetId=spreadsheet_id).execute()
        title = meta.get("properties", {}).get("title", "")
        sheets = meta.get("sheets", [])
        target_sheet = sheet_name or sheets[0]["properties"]["title"] if sheets else "Sheet1"

        range_notation = f"'{target_sheet}'"
        result = (
            spreadsheets.values()
            .get(spreadsheetId=spreadsheet_id, range=range_notation)
            .execute()
        )
        values: list[list[Any]] = result.get("values", [])

        if not values:
            return {
                "spreadsheet_title": title,
                "sheet_name": target_sheet,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "truncated": False,
            }

        max_rows = 500
        headers = values[0]
        data_rows = values[1:max_rows + 1]
        truncated = len(values) - 1 > max_rows

        # Pad short rows to match header length
        col_count = len(headers)
        padded = [row + [""] * (col_count - len(row)) for row in data_rows]

        logger.info(
            "google_sheets.read.done spreadsheet_id=%s rows=%s truncated=%s",
            spreadsheet_id, len(padded), truncated,
        )
        return {
            "spreadsheet_title": title,
            "sheet_name": target_sheet,
            "columns": headers,
            "rows": padded,
            "row_count": len(padded),
            "truncated": truncated,
        }
    except Exception as exc:
        logger.error("google_sheets.read.error spreadsheet_id=%s error=%s", spreadsheet_id, exc, exc_info=True)
        return {"error": str(exc), "error_type": type(exc).__name__}


@tool
def read_google_doc(url: str) -> dict[str, Any]:
    """Read the text content of a Google Doc shared via URL or document ID.

    Call this whenever the user provides a Google Docs link or ID. Returns the
    full plain-text content of the document so you can summarise, answer
    questions, or reference specific sections.

    Args:
        url: Google Docs URL (https://docs.google.com/document/d/...) or
            bare document ID.

    Returns:
        Dict with ``document_title`` and ``content`` (plain text).
        On error: ``{"error": ..., "error_type": ...}``.
    """
    document_id = _extract_id(url, _DOCS_RE)
    if not document_id:
        return {
            "error": f"Could not parse a Google Docs ID from: {url!r}",
            "error_type": "invalid_url",
        }

    logger.info("google_docs.read.start document_id=%s", document_id)
    try:
        from googleapiclient.discovery import build  # type: ignore[import-untyped]

        creds = _build_credentials()
        service = build("docs", "v1", credentials=creds, cache_discovery=False)
        doc = service.documents().get(documentId=document_id).execute()

        title = doc.get("title", "")
        body = doc.get("body", {})
        text = _doc_body_to_text(body)

        logger.info(
            "google_docs.read.done document_id=%s chars=%s", document_id, len(text)
        )
        return {"document_title": title, "content": text}
    except Exception as exc:
        logger.error("google_docs.read.error document_id=%s error=%s", document_id, exc, exc_info=True)
        return {"error": str(exc), "error_type": type(exc).__name__}


GOOGLE_TOOLS = [read_google_sheet, read_google_doc]
