"""
Pydantic models for OACA Invoice Extraction API
"""
from pydantic import BaseModel
from typing import Optional, List


class InvoiceData(BaseModel):
    """Extracted invoice data fields"""
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    client_name: Optional[str] = None
    total_amount: Optional[str] = None
    currency: Optional[str] = None


class ExtractionResponse(BaseModel):
    """API response for invoice extraction"""
    success: bool
    message: str
    data: Optional[InvoiceData] = None
    raw_entities: Optional[dict] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
