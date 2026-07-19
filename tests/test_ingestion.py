import pytest
from unittest.mock import patch, MagicMock
from app.services.ingestion import ingest_folder, ingest_file, _extract_text
import os
from pathlib import Path

@patch.object(Path, "exists")
@patch.object(Path, "iterdir")
@patch("app.services.ingestion.ingest_file")
def test_ingest_folder(mock_ingest_file, mock_iterdir, mock_exists):
    mock_exists.return_value = True
    
    mock_file = MagicMock()
    mock_file.is_file.return_value = True
    mock_file.suffix = ".md"
    mock_file.__str__.return_value = "/docs/test.md"
    
    mock_iterdir.return_value = [mock_file]
    mock_ingest_file.return_value = 5
    
    count = ingest_folder("/docs")
    assert count == 5
    mock_ingest_file.assert_called_once()

@patch.object(Path, "exists")
def test_ingest_missing_folder(mock_exists):
    mock_exists.return_value = False
    with pytest.raises(FileNotFoundError):
        ingest_folder("/docs")

@patch("app.services.ingestion._extract_text")
@patch("app.services.ingestion._split_text")
@patch("app.services.ingestion.ingest_chunks")
def test_ingest_file(mock_ingest, mock_split, mock_extract):
    mock_extract.return_value = "raw text"
    mock_split.return_value = ["chunk1", "chunk2"]
    mock_ingest.return_value = 2

    count = ingest_file("test.pdf")
    assert count == 2
    mock_extract.assert_called_once_with("test.pdf")

@patch("builtins.open")
def test_extract_text_md(mock_open):
    mock_f = MagicMock()
    mock_f.read.return_value = "md content"
    mock_open.return_value.__enter__.return_value = mock_f
    assert _extract_text("file.md") == "md content"

