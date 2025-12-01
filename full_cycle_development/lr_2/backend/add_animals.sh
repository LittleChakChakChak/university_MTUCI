#!/bin/bash

# Функция для добавления животного с паузой
add_animal() {
    local json_data="$1"
    curl -X 'POST' \
      'http://127.0.0.1:8000/animals/' \
      -H 'Content-Type: application/json' \
      -d "$json_data"
    echo ""
    echo "--- Животное добавлено ---"
    sleep 1  # Пауза 1 секунда между запросами
}

echo "Начинаем добавление животных..."

# Животное 1
add_animal '{
  "name": "Барсик",
  "animal_type": "cat",
  "breed": "Дворовый",
  "age": 12,
  "weight": 4.5,
  "description": "Ласковый кот, любит спать на коленях",
  "arrival_date": "2024-01-15",
  "is_vaccinated": true,
  "is_sterilized": true
}'

# Животное 2
add_animal '{
  "name": "Шарик",
  "animal_type": "dog",
  "breed": "Овчарка",
  "age": 24,
  "weight": 25.0,
  "description": "Активная собака, знает команды",
  "arrival_date": "2024-02-10",
  "is_vaccinated": true,
  "is_sterilized": false
}'

# Животное 3
add_animal '{
  "name": "Кеша",
  "animal_type": "bird",
  "breed": "Попугай",
  "age": 6,
  "weight": 0.3,
  "description": "Говорит Привет и Кеша хороший",
  "arrival_date": "2024-03-01",
  "is_vaccinated": false,
  "is_sterilized": false
}'

# Животное 4
add_animal '{
  "name": "Пушистик",
  "animal_type": "rabbit",
  "breed": "Ангорский",
  "age": 8,
  "weight": 1.2,
  "description": "Пушистый кролик, очень спокойный",
  "arrival_date": "2024-02-20",
  "is_vaccinated": true,
  "is_sterilized": true
}'

# Животное 5 (усыновленное)
add_animal '{
  "name": "Рекс",
  "animal_type": "dog",
  "breed": "Лабрадор",
  "age": 18,
  "weight": 28.0,
  "description": "Дружелюбный пес, обожает детей",
  "arrival_date": "2023-12-01",
  "adoption_date": "2024-01-20",
  "status": "adopted",
  "is_vaccinated": true,
  "is_sterilized": true
}'

echo "Все животные добавлены!"