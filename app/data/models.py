from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, Text


Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)
    title = Column(String)
    text = Column(Text)
    image_path = Column(String) # optional path to related image