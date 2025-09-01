from io import BytesIO
from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
import pandas as pd

from ...core.deps import get_db
from ...core.errors import AppException, ErrorCode
from ...schemas.taxonomy import UploadTaxonomyRequest, TaxonomyEntryRequest
from ...models.entities import Taxonomy, TaxonomyEntry

router = APIRouter()


@router.post("/upload")
async def upload_taxonomy(
    file: UploadFile = File(...),
    meta: UploadTaxonomyRequest = Depends(),
    db: Session = Depends(get_db),
):
    contents = await file.read()
    xl = pd.ExcelFile(BytesIO(contents))

    if meta.sheet_name not in xl.sheet_names:
        raise AppException(
            ErrorCode.FILE_VALIDATION_ERROR,
            f"Sheet '{meta.sheet_name}' not found. Available: {xl.sheet_names}",
            status_code=400,
        )

    df = xl.parse(meta.sheet_name)
    df.columns = df.columns.str.lower()
    required_cols = {"tag", "type", "reference"}
    missing = required_cols - set(df.columns)
    if missing:
        raise AppException(
            ErrorCode.FILE_VALIDATION_ERROR,
            f"Missing required columns: {', '.join(missing)}",
            status_code=400,
        )

    taxonomy = Taxonomy(name=meta.sheet_name, symbol=meta.symbol, description=meta.description, source_file=file.filename)
    db.add(taxonomy)
    db.commit()
    db.refresh(taxonomy)

    for _, row in df.iterrows():
        db.add(TaxonomyEntry(taxonomy_id=taxonomy.id, tag=row["tag"], datatype=row["type"], reference=row["reference"]))
    db.commit()

    return {"message": "Taxonomy uploaded successfully", "taxonomy_id": taxonomy.id}


@router.get("/{taxonomy_id}")
def get_taxonomy(taxonomy_id: int, db: Session = Depends(get_db)):
    taxonomy = db.query(Taxonomy).filter(Taxonomy.id == taxonomy_id).first()
    if not taxonomy:
        raise AppException(ErrorCode.NOT_FOUND, "Taxonomy not found", status_code=404)
    return taxonomy


@router.delete("/{taxonomy_id}")
def delete_taxonomy(taxonomy_id: int, db: Session = Depends(get_db)):
    taxonomy = db.query(Taxonomy).filter(Taxonomy.id == taxonomy_id).first()
    if not taxonomy:
        raise AppException(ErrorCode.NOT_FOUND, "Taxonomy not found", status_code=404)
    db.delete(taxonomy)
    db.commit()
    return {"message": "Taxonomy deleted successfully"}


@router.get("/{taxonomy_id}/entries")
def get_entries(taxonomy_id: int, db: Session = Depends(get_db)):
    return db.query(TaxonomyEntry).filter(TaxonomyEntry.taxonomy_id == taxonomy_id).all()


@router.post("/{taxonomy_id}/entries")
def add_entry(taxonomy_id: int, req: TaxonomyEntryRequest, db: Session = Depends(get_db)):
    db.add(TaxonomyEntry(taxonomy_id=taxonomy_id, tag=req.tag, datatype=req.datatype, reference=req.reference))
    db.commit()
    return {"message": "Entry added successfully"}


@router.put("/entries/{entry_id}")
def update_entry(entry_id: int, req: TaxonomyEntryRequest, db: Session = Depends(get_db)):
    entry = db.query(TaxonomyEntry).filter(TaxonomyEntry.id == entry_id).first()
    if not entry:
        raise AppException(ErrorCode.NOT_FOUND, "Entry not found", status_code=404)
    entry.tag, entry.datatype, entry.reference = req.tag, req.datatype, req.reference
    db.commit()
    return {"message": "Entry updated successfully"}


@router.delete("/entries/{entry_id}")
def delete_entry(entry_id: int, db: Session = Depends(get_db)):
    entry = db.query(TaxonomyEntry).filter(TaxonomyEntry.id == entry_id).first()
    if not entry:
        raise AppException(ErrorCode.NOT_FOUND, "Entry not found", status_code=404)
    db.delete(entry)
    db.commit()
    return {"message": "Entry deleted successfully"}
