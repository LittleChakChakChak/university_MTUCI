from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import date
from enum import Enum
from models import AnimalType, AnimalStatus

class AnimalBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    animal_type: AnimalType
    breed: Optional[str] = Field(None, max_length=100)
    age: int = Field(..., ge=0, le=300)  # 0-25 лет в месяцах
    weight: float = Field(..., gt=0, le=200)  # вес в кг
    description: Optional[str] = None
    status: AnimalStatus = AnimalStatus.AVAILABLE
    arrival_date: date
    adoption_date: Optional[date] = None
    is_vaccinated: bool = False
    is_sterilized: bool = False

    @validator('adoption_date')
    def validate_adoption_date(cls, v, values):
        if v and 'arrival_date' in values and v < values['arrival_date']:
            raise ValueError('Дата усыновления не может быть раньше даты поступления')
        return v

class AnimalCreate(AnimalBase):
    pass

class AnimalUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    animal_type: Optional[AnimalType] = None
    breed: Optional[str] = Field(None, max_length=100)
    age: Optional[int] = Field(None, ge=0, le=300)
    weight: Optional[float] = Field(None, gt=0, le=200)
    description: Optional[str] = None
    status: Optional[AnimalStatus] = None
    adoption_date: Optional[date] = None
    is_vaccinated: Optional[bool] = None
    is_sterilized: Optional[bool] = None

class AnimalResponse(AnimalBase):
    id: int

    class Config:
        from_attributes = True

class AnimalStats(BaseModel):
    total_animals: int
    available_count: int
    adopted_count: int
    by_type: dict
    average_age: float