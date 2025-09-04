from sqlalchemy import Column, Integer, String, Boolean, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..db.session import Base


class Setting(Base):
    __tablename__ = "settings"
    id = Column(Integer, primary_key=True, index=True)
    active_embedder_id = Column(Integer, ForeignKey("embedders.id"))
    active_reranker_id = Column(Integer, ForeignKey("rerankers.id"))
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    embedder = relationship("Embedder", back_populates="settings")
    reranker = relationship("Reranker", back_populates="settings")


class Embedder(Base):
    __tablename__ = "embedders"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    version = Column(String)
    path = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    settings = relationship("Setting", back_populates="embedder")


class Reranker(Base):
    __tablename__ = "rerankers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    version = Column(String)
    path = Column(Text)
    normalize_method = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    settings = relationship("Setting", back_populates="reranker")


class Taxonomy(Base):
    __tablename__ = "taxonomies"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    symbol = Column(String, unique=True)
    description = Column(Text)
    source_file = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    entries = relationship("TaxonomyEntry", back_populates="taxonomy")
    feedbacks = relationship("Feedback", back_populates="taxonomy")


class TaxonomyEntry(Base):
    __tablename__ = "taxonomy_entries"
    id = Column(Integer, primary_key=True, index=True)
    taxonomy_id = Column(Integer, ForeignKey("taxonomies.id"))
    tag = Column(String)
    datatype = Column(String)
    reference = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    taxonomy = relationship("Taxonomy", back_populates="entries")


class Feedback(Base):
    __tablename__ = "feedbacks"
    id = Column(Integer, primary_key=True, index=True)
    taxonomy_id = Column(Integer, ForeignKey("taxonomies.id"))
    query = Column(Text)
    reference = Column(Text)
    tag = Column(String)
    is_correct = Column(Boolean)
    is_custom = Column(Boolean, default=False)
    rank = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    taxonomy = relationship("Taxonomy", back_populates="feedbacks")
