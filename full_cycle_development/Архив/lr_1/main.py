from fastapi import FastAPI, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import date
import models
import schemas
from database import engine, get_db
from sqlalchemy import func

# Создание таблиц
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Animal Shelter API",
    description="REST API для управления приютом животных",
    version="1.0.0"
)


# Вспомогательные функции
def get_animal_or_404(db: Session, animal_id: int):
    animal = db.query(models.Animal).filter(models.Animal.id == animal_id).first()
    if not animal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Животное не найдено"
        )
    return animal


# Эндпоинты
@app.post("/animals/", response_model=schemas.AnimalResponse, status_code=status.HTTP_201_CREATED)
def create_animal(animal: schemas.AnimalCreate, db: Session = Depends(get_db)):
    """Создание нового животного в приюте"""
    db_animal = models.Animal(**animal.dict())
    db.add(db_animal)
    db.commit()
    db.refresh(db_animal)
    return db_animal


@app.get("/animals/", response_model=List[schemas.AnimalResponse])
def get_animals(
        db: Session = Depends(get_db),
        skip: int = Query(0, ge=0, description="Количество записей для пропуска"),
        limit: int = Query(100, ge=1, le=1000, description="Лимит записей"),
        animal_type: Optional[models.AnimalType] = Query(None, description="Фильтр по виду животного"),
        status: Optional[models.AnimalStatus] = Query(None, description="Фильтр по статусу"),
        search: Optional[str] = Query(None, description="Поиск по имени или породе"),
        sort_by: str = Query("id", description="Поле для сортировки"),
        sort_order: str = Query("asc", regex="^(asc|desc)$", description="Порядок сортировки")
):
    """Получение списка животных с фильтрацией, поиском и сортировкой"""
    query = db.query(models.Animal)

    # Фильтрация по виду
    if animal_type:
        query = query.filter(models.Animal.animal_type == animal_type)

    # Фильтрация по статусу
    if status:
        query = query.filter(models.Animal.status == status)

    # Поиск по имени или породе
    if search:
        query = query.filter(
            (models.Animal.name.ilike(f"%{search}%")) |
            (models.Animal.breed.ilike(f"%{search}%"))
        )

    # Сортировка
    sort_column = getattr(models.Animal, sort_by, models.Animal.id)
    if sort_order == "desc":
        sort_column = sort_column.desc()
    query = query.order_by(sort_column)

    # Пагинация
    animals = query.offset(skip).limit(limit).all()
    return animals


@app.get("/animals/{animal_id}", response_model=schemas.AnimalResponse)
def get_animal(animal_id: int, db: Session = Depends(get_db)):
    """Получение информации о конкретном животном"""
    return get_animal_or_404(db, animal_id)


@app.put("/animals/{animal_id}", response_model=schemas.AnimalResponse)
def update_animal(animal_id: int, animal_update: schemas.AnimalUpdate, db: Session = Depends(get_db)):
    """Обновление информации о животном"""
    db_animal = get_animal_or_404(db, animal_id)

    update_data = animal_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_animal, field, value)

    db.commit()
    db.refresh(db_animal)
    return db_animal


@app.delete("/animals/{animal_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_animal(animal_id: int, db: Session = Depends(get_db)):
    """Удаление животного из базы данных"""
    db_animal = get_animal_or_404(db, animal_id)
    db.delete(db_animal)
    db.commit()
    return None


@app.get("/animals/{animal_id}/adopt", response_model=schemas.AnimalResponse)
def adopt_animal(animal_id: int, db: Session = Depends(get_db)):
    """Отметить животное как усыновленное"""
    db_animal = get_animal_or_404(db, animal_id)

    if db_animal.status == models.AnimalStatus.ADOPTED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Животное уже усыновлено"
        )

    db_animal.status = models.AnimalStatus.ADOPTED
    db_animal.adoption_date = date.today()

    db.commit()
    db.refresh(db_animal)
    return db_animal


@app.get("/animals/{animal_id}/medical", response_model=schemas.AnimalResponse)
def move_to_medical_care(animal_id: int, db: Session = Depends(get_db)):
    """Перевести животное на медицинский уход"""
    db_animal = get_animal_or_404(db, animal_id)
    db_animal.status = models.AnimalStatus.MEDICAL
    db.commit()
    db.refresh(db_animal)
    return db_animal


@app.get("/stats/", response_model=schemas.AnimalStats)
def get_shelter_stats(db: Session = Depends(get_db)):
    """Получение статистики по приюту"""
    # Общее количество животных
    total_animals = db.query(models.Animal).count()

    # Количество по статусам
    available_count = db.query(models.Animal).filter(
        models.Animal.status == models.AnimalStatus.AVAILABLE
    ).count()

    adopted_count = db.query(models.Animal).filter(
        models.Animal.status == models.AnimalStatus.ADOPTED
    ).count()

    # Количество по видам
    type_counts = db.query(
        models.Animal.animal_type,
        func.count(models.Animal.id)
    ).group_by(models.Animal.animal_type).all()

    by_type = {animal_type.value: count for animal_type, count in type_counts}

    # Средний возраст
    average_age = db.query(func.avg(models.Animal.age)).scalar() or 0

    return schemas.AnimalStats(
        total_animals=total_animals,
        available_count=available_count,
        adopted_count=adopted_count,
        by_type=by_type,
        average_age=round(average_age, 2)
    )


@app.get("/animals/types/{animal_type}/count")
def get_animal_type_count(animal_type: models.AnimalType, db: Session = Depends(get_db)):
    """Получение количества животных определенного вида"""
    count = db.query(models.Animal).filter(models.Animal.animal_type == animal_type).count()
    return {"animal_type": animal_type, "count": count}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)