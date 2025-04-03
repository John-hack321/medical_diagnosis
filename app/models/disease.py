from sqlalchemy import Column, Integer, String, Text, Float, ForeignKey, Table
from sqlalchemy.orm import relationship
from app.database import Base

# Association table for many-to-many relationship
disease_symptoms = Table(
    'disease_symptoms',
    Base.metadata,
    Column('disease_id', Integer, ForeignKey('diseases.id'), primary_key=True),
    Column('symptom_id', Integer, ForeignKey('symptoms.id'), primary_key=True),
    Column('confidence', Float, default=0.5)  # Confidence score for this association
)

class Disease(Base):
    __tablename__ = 'diseases'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    treatment = Column(Text)
    prevention = Column(Text)
    risk_factors = Column(Text)
    epidemiology = Column(Text)
    source_url = Column(String(500))
    scrape_date = Column(String(50))
    
    # Relationship with symptoms (many-to-many)
    symptoms = relationship('Symptom', secondary='disease_symptoms', back_populates='diseases')
    
class Symptom(Base):
    __tablename__ = 'symptoms'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    
    # Relationship with diseases (many-to-many)
    diseases = relationship('Disease', secondary='disease_symptoms', back_populates='symptoms')