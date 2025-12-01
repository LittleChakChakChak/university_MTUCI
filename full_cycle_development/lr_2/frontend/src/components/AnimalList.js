import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { animalApi } from '../services/api';
import AnimalFilter from './AnimalFilter';

const AnimalList = () => {
    const [animals, setAnimals] = useState([]);
    const [filteredAnimals, setFilteredAnimals] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [notification, setNotification] = useState(null);

    useEffect(() => {
        loadAnimals();
    }, []);

    const loadAnimals = async () => {
        try {
            setLoading(true);
            const response = await animalApi.getAllAnimals();
            const animalsData = response.data;
            setAnimals(animalsData);
            setFilteredAnimals(animalsData);
            setError('');
        } catch (error) {
            console.error('Error loading animals:', error);
            setError('Не удалось загрузить список животных');
        } finally {
            setLoading(false);
        }
    };

    // Функция для применения фильтров
    const applyFilters = (filters) => {
        let filtered = [...animals];

        // Поиск по имени или породе
        if (filters.search) {
            const searchLower = filters.search.toLowerCase();
            filtered = filtered.filter(animal =>
                animal.name.toLowerCase().includes(searchLower) ||
                (animal.breed && animal.breed.toLowerCase().includes(searchLower))
            );
        }

        // Фильтр по типу животного
        if (filters.animal_type) {
            filtered = filtered.filter(animal => animal.animal_type === filters.animal_type);
        }

        // Фильтр по статусу
        if (filters.status) {
            filtered = filtered.filter(animal => animal.status === filters.status);
        }

        // Фильтр по возрасту
        if (filters.minAge) {
            filtered = filtered.filter(animal => animal.age >= parseInt(filters.minAge));
        }
        if (filters.maxAge) {
            filtered = filtered.filter(animal => animal.age <= parseInt(filters.maxAge));
        }

        // Фильтр по вакцинации
        if (filters.vaccinated === 'true') {
            filtered = filtered.filter(animal => animal.is_vaccinated);
        } else if (filters.vaccinated === 'false') {
            filtered = filtered.filter(animal => !animal.is_vaccinated);
        }

        // Фильтр по стерилизации
        if (filters.sterilized === 'true') {
            filtered = filtered.filter(animal => animal.is_sterilized);
        } else if (filters.sterilized === 'false') {
            filtered = filtered.filter(animal => !animal.is_sterilized);
        }

        setFilteredAnimals(filtered);
    };

    const handleDelete = async (id, name) => {
        if (window.confirm(`Вы уверены, что хотите удалить животное "${name}"?`)) {
            try {
                await animalApi.deleteAnimal(id);
                loadAnimals();
                showNotification('success', 'Животное успешно удалено');
            } catch (error) {
                console.error('Error deleting animal:', error);
                showNotification('error', 'Ошибка при удалении животного');
            }
        }
    };

    // Функция для усыновления
    const handleAdopt = async (id, name) => {
        if (!window.confirm(`Вы уверены, что хотите отметить животное "${name}" как усыновленное?`)) {
            return;
        }

        try {
            // Отправляем GET запрос на эндпоинт усыновления
            const response = await animalApi.adoptAnimal(id);

            // Обновляем локальное состояние
            const updatedAnimals = animals.map(animal => {
                if (animal.id === id) {
                    return {
                        ...animal,
                        status: 'adopted',
                        adoption_date: new Date().toISOString().split('T')[0]
                    };
                }
                return animal;
            });

            setAnimals(updatedAnimals);
            applyFilters({}); // Применяем текущие фильтры

            showNotification('success', `Животное "${name}" отмечено как усыновленное!`);

        } catch (error) {
            console.error('Error adopting animal:', error);

            let errorMessage = 'Ошибка при усыновлении животного';

            if (error.response) {
                // Сервер ответил с ошибкой
                if (error.response.status === 400) {
                    errorMessage = 'Животное уже усыновлено';
                } else if (error.response.status === 404) {
                    errorMessage = 'Животное не найдено';
                } else if (error.response.data && error.response.data.detail) {
                    errorMessage = `Ошибка: ${error.response.data.detail}`;
                }
            } else if (error.request) {
                // Запрос был сделан, но ответа не было
                errorMessage = 'Нет ответа от сервера. Проверьте подключение.';
            }

            showNotification('error', errorMessage);
        }
    };

    // Функция для уведомлений
    const showNotification = (type, message) => {
        setNotification({ type, message });
        setTimeout(() => setNotification(null), 3000);
    };

    if (loading) {
        return (
            <div style={{ textAlign: 'center', padding: '40px' }}>
                <div style={{ fontSize: '24px', marginBottom: '20px' }}>⏳</div>
                <p>Загрузка животных...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div style={{
                padding: '20px',
                backgroundColor: '#ffebee',
                color: '#c62828',
                borderRadius: '8px',
                marginBottom: '20px',
            }}>
                <p>{error}</p>
                <button
                    onClick={loadAnimals}
                    style={{
                        padding: '8px 16px',
                        backgroundColor: '#2196F3',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                    }}
                >
                    Попробовать снова
                </button>
            </div>
        );
    }

    const getTypeEmoji = (type) => {
        switch (type) {
            case 'dog': return '🐕';
            case 'cat': return '🐈';
            case 'bird': return '🐦';
            case 'rabbit': return '🐇';
            default: return '🐾';
        }
    };

    const getStatusBadge = (status) => {
        const statusColors = {
            available: { bg: '#d4edda', color: '#155724', text: 'Доступен' },
            adopted: { bg: '#d1ecf1', color: '#0c5460', text: 'Усыновлен' },
            medical_care: { bg: '#fff3cd', color: '#856404', text: 'Мед. уход' },
            foster_care: { bg: '#e2d9f3', color: '#4a2d7c', text: 'Передержка' },
            quarantine: { bg: '#f8d7da', color: '#721c24', text: 'Карантин' },
        };

        const statusInfo = statusColors[status] || { bg: '#e9ecef', color: '#495057', text: status };

        return (
            <span style={{
                backgroundColor: statusInfo.bg,
                color: statusInfo.color,
                padding: '4px 8px',
                borderRadius: '12px',
                fontSize: '12px',
                display: 'inline-block',
                fontWeight: '500',
                border: `1px solid ${statusInfo.color}20`,
            }}>
                {statusInfo.text}
            </span>
        );
    };

    return (
        <div>
            {/* Уведомления */}
            {notification && (
                <div style={{
                    position: 'fixed',
                    top: '20px',
                    right: '20px',
                    zIndex: 1000,
                    padding: '15px 20px',
                    backgroundColor: notification.type === 'success' ? '#d4edda' : '#f8d7da',
                    color: notification.type === 'success' ? '#155724' : '#721c24',
                    borderRadius: '8px',
                    border: `1px solid ${notification.type === 'success' ? '#c3e6cb' : '#f5c6cb'}`,
                    boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    minWidth: '300px',
                }}>
                    <span style={{ fontSize: '20px' }}>
                        {notification.type === 'success' ? '✅' : '❌'}
                    </span>
                    <span>{notification.message}</span>
                    <button
                        onClick={() => setNotification(null)}
                        style={{
                            marginLeft: 'auto',
                            background: 'none',
                            border: 'none',
                            fontSize: '18px',
                            cursor: 'pointer',
                            color: 'inherit',
                        }}
                    >
                        ×
                    </button>
                </div>
            )}

            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '20px',
            }}>
                <h2 style={{ margin: 0 }}>
                    🐾 Животные в приюте
                    <span style={{ fontSize: '16px', color: '#6c757d', marginLeft: '10px' }}>
                        ({filteredAnimals.length} из {animals.length})
                    </span>
                </h2>
                <Link to="/add-animal">
                    <button style={{
                        padding: '10px 20px',
                        backgroundColor: '#28a745',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontSize: '16px',
                        fontWeight: '500',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                    }}>
                        <span>+</span> Добавить животное
                    </button>
                </Link>
            </div>

            {/* Компонент фильтрации */}
            <AnimalFilter onFilter={applyFilters} animals={animals} />

            {/* Статистика фильтрации */}
            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '15px',
                padding: '10px 15px',
                backgroundColor: '#e9f7fe',
                borderRadius: '6px',
                border: '1px solid #b8e6ff',
            }}>
                <div>
                    <strong>Найдено животных:</strong> {filteredAnimals.length}
                    {filteredAnimals.length !== animals.length && (
                        <span style={{ color: '#6c757d', marginLeft: '10px' }}>
                            (отфильтровано из {animals.length})
                        </span>
                    )}
                </div>
                <div style={{ display: 'flex', gap: '10px' }}>
                    <button
                        onClick={() => {
                            const availableAnimals = animals.filter(a => a.status === 'available');
                            applyFilters({ status: 'available' });
                            showNotification('info', `Показаны только доступные животные: ${availableAnimals.length}`);
                        }}
                        style={{
                            padding: '6px 12px',
                            backgroundColor: '#17a2b8',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '14px',
                        }}
                    >
                        Показать доступных
                    </button>
                    <button
                        onClick={loadAnimals}
                        style={{
                            padding: '6px 12px',
                            backgroundColor: '#6c757d',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '14px',
                        }}
                    >
                        Обновить список
                    </button>
                </div>
            </div>

            {filteredAnimals.length === 0 ? (
                <div style={{
                    textAlign: 'center',
                    padding: '40px',
                    backgroundColor: '#f8f9fa',
                    borderRadius: '8px',
                    border: '2px dashed #dee2e6',
                }}>
                    <div style={{ fontSize: '48px', marginBottom: '20px' }}>🔍</div>
                    <h3 style={{ color: '#6c757d' }}>Животные не найдены</h3>
                    <p style={{ color: '#868e96' }}>
                        Попробуйте изменить параметры фильтрации или добавьте новое животное
                    </p>
                    <Link to="/add-animal">
                        <button style={{
                            padding: '10px 20px',
                            backgroundColor: '#28a745',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            marginTop: '20px',
                        }}>
                            Добавить животное
                        </button>
                    </Link>
                </div>
            ) : (
                <div style={{
                    overflowX: 'auto',
                    backgroundColor: 'white',
                    borderRadius: '8px',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                }}>
                    <table style={{
                        width: '100%',
                        borderCollapse: 'collapse',
                        minWidth: '800px',
                    }}>
                        <thead>
                            <tr style={{ backgroundColor: '#f8f9fa' }}>
                                <th style={{
                                    padding: '12px 16px',
                                    textAlign: 'left',
                                    borderBottom: '2px solid #dee2e6',
                                    fontWeight: '600',
                                    color: '#495057',
                                }}>#</th>
                                <th style={{
                                    padding: '12px 16px',
                                    textAlign: 'left',
                                    borderBottom: '2px solid #dee2e6',
                                    fontWeight: '600',
                                    color: '#495057',
                                }}>Животное</th>
                                <th style={{
                                    padding: '12px 16px',
                                    textAlign: 'left',
                                    borderBottom: '2px solid #dee2e6',
                                    fontWeight: '600',
                                    color: '#495057',
                                }}>Порода</th>
                                <th style={{
                                    padding: '12px 16px',
                                    textAlign: 'left',
                                    borderBottom: '2px solid #dee2e6',
                                    fontWeight: '600',
                                    color: '#495057',
                                }}>Возраст</th>
                                <th style={{
                                    padding: '12px 16px',
                                    textAlign: 'left',
                                    borderBottom: '2px solid #dee2e6',
                                    fontWeight: '600',
                                    color: '#495057',
                                }}>Вес</th>
                                <th style={{
                                    padding: '12px 16px',
                                    textAlign: 'left',
                                    borderBottom: '2px solid #dee2e6',
                                    fontWeight: '600',
                                    color: '#495057',
                                }}>Статус</th>
                                <th style={{
                                    padding: '12px 16px',
                                    textAlign: 'left',
                                    borderBottom: '2px solid #dee2e6',
                                    fontWeight: '600',
                                    color: '#495057',
                                }}>Дата поступления</th>
                                <th style={{
                                    padding: '12px 16px',
                                    textAlign: 'left',
                                    borderBottom: '2px solid #dee2e6',
                                    fontWeight: '600',
                                    color: '#495057',
                                }}>Действия</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredAnimals.map((animal, index) => (
                                <tr key={animal.id} style={{
                                    borderBottom: '1px solid #eee',
                                    backgroundColor: index % 2 === 0 ? '#fcfcfc' : 'white',
                                    transition: 'background-color 0.2s',
                                }}>
                                    <td style={{ padding: '12px 16px', fontWeight: '500' }}>{animal.id}</td>
                                    <td style={{ padding: '12px 16px' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                            <span style={{ fontSize: '24px' }}>
                                                {getTypeEmoji(animal.animal_type)}
                                            </span>
                                            <div>
                                                <strong style={{ fontSize: '16px' }}>{animal.name}</strong>
                                                <div style={{
                                                    fontSize: '13px',
                                                    color: '#6c757d',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    gap: '8px',
                                                    marginTop: '2px',
                                                }}>
                                                    <span>{animal.animal_type}</span>
                                                    {animal.is_vaccinated && (
                                                        <span title="Вакцинирован" style={{ color: '#28a745' }}>💉</span>
                                                    )}
                                                    {animal.is_sterilized && (
                                                        <span title="Стерилизован" style={{ color: '#dc3545' }}>✂️</span>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    </td>
                                    <td style={{ padding: '12px 16px' }}>
                                        {animal.breed || (
                                            <span style={{ color: '#6c757d', fontStyle: 'italic' }}>—</span>
                                        )}
                                    </td>
                                    <td style={{ padding: '12px 16px' }}>
                                        <div>
                                            <span style={{ fontWeight: '500' }}>{animal.age} мес.</span>
                                            <div style={{ fontSize: '12px', color: '#868e96' }}>
                                                (~{(animal.age / 12).toFixed(1)} лет)
                                            </div>
                                        </div>
                                    </td>
                                    <td style={{ padding: '12px 16px', fontWeight: '500' }}>
                                        {animal.weight} кг
                                    </td>
                                    <td style={{ padding: '12px 16px' }}>
                                        {getStatusBadge(animal.status)}
                                    </td>
                                    <td style={{ padding: '12px 16px' }}>
                                        <div style={{ fontSize: '14px' }}>
                                            {new Date(animal.arrival_date).toLocaleDateString('ru-RU')}
                                        </div>
                                        <div style={{ fontSize: '12px', color: '#868e96' }}>
                                            {Math.floor((new Date() - new Date(animal.arrival_date)) / (1000 * 60 * 60 * 24))} дней назад
                                        </div>
                                    </td>
                                    <td style={{ padding: '12px 16px' }}>
                                        <div style={{ display: 'flex', gap: '8px' }}>
                                            {animal.status === 'available' ? (
                                                <button
                                                    onClick={() => handleAdopt(animal.id, animal.name)}
                                                    title="Отметить как усыновленное"
                                                    style={{
                                                        padding: '6px 12px',
                                                        backgroundColor: '#17a2b8',
                                                        color: 'white',
                                                        border: 'none',
                                                        borderRadius: '4px',
                                                        cursor: 'pointer',
                                                        fontSize: '12px',
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        gap: '4px',
                                                    }}
                                                >
                                                    🏠 Усыновить
                                                </button>
                                            ) : (
                                                <span style={{
                                                    padding: '6px 12px',
                                                    backgroundColor: '#6c757d',
                                                    color: 'white',
                                                    borderRadius: '4px',
                                                    fontSize: '12px',
                                                }}>
                                                    Уже усыновлен
                                                </span>
                                            )}
                                            <button
                                                onClick={() => handleDelete(animal.id, animal.name)}
                                                title="Удалить животное"
                                                style={{
                                                    padding: '6px 12px',
                                                    backgroundColor: '#dc3545',
                                                    color: 'white',
                                                    border: 'none',
                                                    borderRadius: '4px',
                                                    cursor: 'pointer',
                                                    fontSize: '12px',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    gap: '4px',
                                                }}
                                            >
                                                🗑️ Удалить
                                            </button>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {/* Информация внизу */}
            <div style={{
                marginTop: '20px',
                padding: '15px',
                backgroundColor: '#f8f9fa',
                borderRadius: '8px',
                fontSize: '14px',
                color: '#6c757d',
            }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <div>
                        <strong>Статистика:</strong>
                        <span style={{ marginLeft: '10px' }}>
                            Всего: {animals.length}
                        </span>
                        <span style={{ marginLeft: '10px' }}>
                            Доступны: {animals.filter(a => a.status === 'available').length}
                        </span>
                        <span style={{ marginLeft: '10px' }}>
                            Усыновлены: {animals.filter(a => a.status === 'adopted').length}
                        </span>
                    </div>
                    <div>
                        Последнее обновление: {new Date().toLocaleTimeString('ru-RU')}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AnimalList;