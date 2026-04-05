from typing import List, Optional

from pydantic import BaseModel, Field


class BddkSearchRequest(BaseModel):
    """
    Request model for searching BDDK decisions.

    BDDK (Bankacilik Duzenleme ve Denetleme Kurumu) is Turkey's Banking
    Regulation and Supervision Agency.
    """
    keywords: str = Field(..., description="Search keywords in Turkish")
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(10, ge=1, le=50, description="Results per page (1-50)")
    category: Optional[str] = Field(None, description="Filter by category (e.g. 'Yönetmelik', 'Genelge', 'Kurul Kararı')")
    date_from: Optional[str] = Field(None, description="Filter from date (DD.MM.YYYY)")
    date_to: Optional[str] = Field(None, description="Filter to date (DD.MM.YYYY)")


class BddkDecisionSummary(BaseModel):
    """Summary of a BDDK decision from search results."""
    title: str = Field(..., description="Decision title")
    document_id: str = Field(..., description="BDDK document ID (e.g., '310' or 'mevzuat_42628')")
    content: str = Field("", description="Decision summary/excerpt")
    decision_date: str = Field("", description="Decision date (e.g., '12.03.2026')")
    decision_number: str = Field("", description="Decision number (e.g., '11428')")
    category: str = Field("", description="Document category (e.g., 'Yönetmelik', 'Genelge', 'Kurul Kararı')")
    source_url: str = Field("", description="Full URL to the document source")


class BddkSearchResult(BaseModel):
    """Response model for BDDK decision search results."""
    decisions: List[BddkDecisionSummary] = Field(
        default_factory=list,
        description="List of matching BDDK decisions",
    )
    total_results: int = Field(0, description="Total number of results")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(10, description="Results per page")


class BddkDocumentMarkdown(BaseModel):
    """
    BDDK decision document converted to Markdown format.

    Supports paginated content for long documents (5000 chars per page).
    """
    document_id: str = Field(..., description="BDDK document ID")
    markdown_content: str = Field("", description="Document content in Markdown")
    page_number: int = Field(1, description="Current page number")
    total_pages: int = Field(1, description="Total number of pages")
