# Animal Shelter Management System

## Описание проекта
Веб-приложение для управления приютом животных с React фронтендом и FastAPI бэкендом.

## Функциональность
- Просмотр, добавление, редактирование и удаление животных
- Фильтрация и поиск по различным параметрам
- Отметка животных как усыновленных
- Статистика и аналитика приюта

## Технологии
- **Frontend**: React, Axios, React Router
- **Backend**: FastAPI, SQLAlchemy, Pydantic
- **База данных**: SQLite

## Установка и запуск

### Бэкенд:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Фронтенд:
```bash
cd frontend
npm install
npm start
```