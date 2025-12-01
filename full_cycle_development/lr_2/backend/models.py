from sqlalchemy import Column, Integer, String, Text, Date, Enum, Boolean
from database import Base
import enum

class AnimalType(str, enum.Enum):
    DOG = "dog"
    CAT = "cat"
    BIRD = "bird"
    RABBIT = "rabbit"
    OTHER = "other"

class AnimalStatus(str, enum.Enum):
    AVAILABLE = "available"
    ADOPTED = "adopted"
    FOSTER = "foster_care"
    MEDICAL = "medical_care"
    QUARANTINE = "quarantine"

class Animal(Base):
    __tablename__ = "animals"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    animal_type = Column(Enum(AnimalType), nullable=False)
    breed = Column(String(100))
    age = Column(Integer)  # в месяцах
    weight = Column(Integer)  # в кг
    description = Column(Text)
    status = Column(Enum(AnimalStatus), default=AnimalStatus.AVAILABLE)
    arrival_date = Column(Date, nullable=False)
    adoption_date = Column(Date, nullable=True)
    is_vaccinated = Column(Boolean, default=False)
    is_sterilized = Column(Boolean, default=False)