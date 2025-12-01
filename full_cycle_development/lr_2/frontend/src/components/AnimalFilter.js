import React, { useState } from 'react';

const AnimalFilter = ({ onFilter, animals = [] }) => {
    const [filters, setFilters] = useState({
        animal_type: '',
        status: '',
        search: '',
        minAge: '',
        maxAge: '',
        vaccinated: '',
        sterilized: '',
    });

    // Получаем уникальные типы животных из списка
    const animalTypes = [...new Set(animals.map(a => a.animal_type))].filter(Boolean);
    const statuses = [...new Set(animals.map(a => a.status))].filter(Boolean);

    const handleChange = (e) => {
        const { name, value, type, checked } = e.target;
        const newFilters = {
            ...filters,
            [name]: type === 'checkbox' ? checked : value,
        };
        setFilters(newFilters);
        onFilter(newFilters);
    };

    const handleClear = () => {
        const clearedFilters = {
            animal_type: '',
            status: '',
            search: '',
            minAge: '',
            maxAge: '',
            vaccinated: '',
            sterilized: '',
        };
        setFilters(clearedFilters);
        onFilter(clearedFilters);
    };

    return (
        <div style={{
            backgroundColor: '#f8f9fa',
            padding: '20px',
            borderRadius: '8px',
            marginBottom: '20px',
            border: '1px solid #dee2e6',
        }}>
            <h3 style={{ marginTop: 0, marginBottom: '15px', color: '#495057' }}>
                🔍 Фильтры животных
            </h3>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '15px' }}>
                {/* Поиск по имени/породе */}
                <div style={{ flex: '1 1 300px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: '500' }}>
                        Поиск по имени или породе
                    </label>
                    <input
                        type="text"
                        name="search"
                        value={filters.search}
                        onChange={handleChange}
                        placeholder="Введите имя или породу..."
                        style={{
                            width: '100%',
                            padding: '8px 12px',
                            border: '1px solid #ced4da',
                            borderRadius: '4px',
                            fontSize: '14px',
                        }}
                    />
                </div>

                {/* Фильтр по типу животного */}
                <div style={{ flex: '1 1 200px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: '500' }}>
                        Вид животного
                    </label>
                    <select
                        name="animal_type"
                        value={filters.animal_type}
                        onChange={handleChange}
                        style={{
                            width: '100%',
                            padding: '8px 12px',
                            border: '1px solid #ced4da',
                            borderRadius: '4px',
                            fontSize: '14px',
                            backgroundColor: 'white',
                        }}
                    >
                        <option value="">Все виды</option>
                        {animalTypes.map(type => (
                            <option key={type} value={type}>
                                {type === 'dog' ? 'Собака 🐕' :
                                 type === 'cat' ? 'Кошка 🐈' :
                                 type === 'bird' ? 'Птица 🐦' :
                                 type === 'rabbit' ? 'Кролик 🐇' : type}
                            </option>
                        ))}
                    </select>
                </div>

                {/* Фильтр по статусу */}
                <div style={{ flex: '1 1 200px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: '500' }}>
                        Статус
                    </label>
                    <select
                        name="status"
                        value={filters.status}
                        onChange={handleChange}
                        style={{
                            width: '100%',
                            padding: '8px 12px',
                            border: '1px solid #ced4da',
                            borderRadius: '4px',
                            fontSize: '14px',
                            backgroundColor: 'white',
                        }}
                    >
                        <option value="">Все статусы</option>
                        {statuses.map(status => (
                            <option key={status} value={status}>
                                {status === 'available' ? 'Доступен ✅' :
                                 status === 'adopted' ? 'Усыновлен 🏠' :
                                 status === 'medical_care' ? 'Мед. уход 🏥' :
                                 status === 'foster_care' ? 'Передержка 🏡' :
                                 status === 'quarantine' ? 'Карантин 🚫' : status}
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            {/* Вторая строка фильтров */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '15px', marginTop: '15px' }}>
                {/* Фильтр по возрасту */}
                <div style={{ flex: '1 1 200px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: '500' }}>
                        Возраст (месяцы)
                    </label>
                    <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                        <input
                            type="number"
                            name="minAge"
                            value={filters.minAge}
                            onChange={handleChange}
                            placeholder="От"
                            min="0"
                            style={{
                                flex: 1,
                                padding: '8px 12px',
                                border: '1px solid #ced4da',
                                borderRadius: '4px',
                                fontSize: '14px',
                            }}
                        />
                        <span style={{ color: '#6c757d' }}>—</span>
                        <input
                            type="number"
                            name="maxAge"
                            value={filters.maxAge}
                            onChange={handleChange}
                            placeholder="До"
                            min="0"
                            style={{
                                flex: 1,
                                padding: '8px 12px',
                                border: '1px solid #ced4da',
                                borderRadius: '4px',
                                fontSize: '14px',
                            }}
                        />
                    </div>
                </div>

                {/* Фильтр по вакцинации */}
                <div style={{ flex: '1 1 150px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: '500' }}>
                        Вакцинация
                    </label>
                    <select
                        name="vaccinated"
                        value={filters.vaccinated}
                        onChange={handleChange}
                        style={{
                            width: '100%',
                            padding: '8px 12px',
                            border: '1px solid #ced4da',
                            borderRadius: '4px',
                            fontSize: '14px',
                            backgroundColor: 'white',
                        }}
                    >
                        <option value="">Все</option>
                        <option value="true">Вакцинированы 💉</option>
                        <option value="false">Не вакцинированы</option>
                    </select>
                </div>

                {/* Фильтр по стерилизации */}
                <div style={{ flex: '1 1 150px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: '500' }}>
                        Стерилизация
                    </label>
                    <select
                        name="sterilized"
                        value={filters.sterilized}
                        onChange={handleChange}
                        style={{
                            width: '100%',
                            padding: '8px 12px',
                            border: '1px solid #ced4da',
                            borderRadius: '4px',
                            fontSize: '14px',
                            backgroundColor: 'white',
                        }}
                    >
                        <option value="">Все</option>
                        <option value="true">Стерилизованы ✂️</option>
                        <option value="false">Не стерилизованы</option>
                    </select>
                </div>

                {/* Кнопки */}
                <div style={{ flex: '1 1 100px', display: 'flex', alignItems: 'flex-end' }}>
                    <button
                        onClick={handleClear}
                        style={{
                            width: '100%',
                            padding: '8px 16px',
                            backgroundColor: '#6c757d',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '14px',
                            fontWeight: '500',
                        }}
                    >
                        Сбросить
                    </button>
                </div>
            </div>
        </div>
    );
};

export default AnimalFilter;