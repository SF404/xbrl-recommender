from pydantic import BaseModel
from typing import Optional


class UploadTaxonomyRequest(BaseModel):
    symbol: str
    description: str
    sheet_name: str


class TaxonomyEntryRequest(BaseModel):
    tag: str
    datatype: str
    reference: str
