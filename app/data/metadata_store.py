import os
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from .models import Base, Document


class MetadataStore:
    def __init__(self, sqlite_url: str):
        os.makedirs(os.path.dirname(sqlite_url.replace("sqlite:///", "")), exist_ok=True)
        self.engine = create_engine(sqlite_url, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, future=True)

    def upsert_document(self, doc_id: str, title: str, text: str, image_path: str | None):
        with self.Session() as s:
            obj = s.get(Document, doc_id)
            if not obj:
                obj = Document(id=doc_id, title=title, text=text, image_path=image_path)
                s.add(obj)
            else:
                obj.title, obj.text, obj.image_path = title, text, image_path
            s.commit()

    def get_texts_by_ids(self, ids: list[str]) -> list[Document]:
        with self.Session() as s:
            q = s.execute(select(Document).where(Document.id.in_(ids)))
            return [row[0] for row in q.all()]