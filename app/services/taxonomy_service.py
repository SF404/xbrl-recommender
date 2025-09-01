from typing import List
from sqlalchemy.orm import Session
from ..models.entities import Taxonomy, TaxonomyEntry


def get_taxonomy_by_symbol(db: Session, symbol: str) -> Taxonomy:
    return db.query(Taxonomy).filter(Taxonomy.symbol == symbol).first()


def get_taxonomy(db: Session, taxonomy_id: int) -> Taxonomy:
    return db.query(Taxonomy).filter(Taxonomy.id == taxonomy_id).first()


def add_taxonomy_entry(db: Session, taxonomy_id: int, tag: str, datatype: str, reference: str):
    entry = TaxonomyEntry(taxonomy_id=taxonomy_id, tag=tag, datatype=datatype, reference=reference)
    db.add(entry)
    db.commit()
    return entry
