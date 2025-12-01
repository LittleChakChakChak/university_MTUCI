import React, { useState } from 'react';
import { animalApi } from '../services/api';

const AddAnimalForm = ({ onAnimalAdded }) => {
    const [formData, setFormData] = useState({
        name: '',
        animal_type: 'dog',
        breed: '',
        age: '',
        weight: '',
        description: '',
        status: 'available',
        arrival_date: new Date().toISOString().split('T')[0],
        is_vaccinated: false,
        is_sterilized: false,
    });

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    const animalTypes = [
        { value: 'dog', label: 'Собака' },
        { value: 'cat', label: 'Кошка' },
        { value: 'bird', label: 'Птица' },
        { value: 'rabbit', label: 'Кролик' },
        { value: 'other', label: 'Другое' },
    ];

    const statusOptions = [
        { value: 'available', label: 'Доступен для усыновления' },
        { value: 'foster_care', label: 'Передержка' },
        { value: 'medical_care', label: 'Медицинский уход' },
        { value: 'quarantine', label: 'Карантин' },
    ];

    const handleChange = (e) => {
        const { name, value, type, checked } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : value,
        }));
    };

    const validateForm = () => {
        if (!formData.name.trim()) {
            setError('Имя животного обязательно');
            return false;
        }
        if (!formData.age || formData.age < 0) {
            setError('Возраст должен быть положительным числом');
            return false;
        }
        if (!formData.weight || formData.weight <= 0) {
            setError('Вес должен быть больше 0');
            return false;
        }
        if (!formData.arrival_date) {
            setError('Дата поступления обязательна');
            return false;
        }
        return true;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setSuccess('');

        if (!validateForm()) {
            return;
        }

        setLoading(true);
        try {
            // Преобразуем строковые числа в числа
            const dataToSend = {
                ...formData,
                age: parseInt(formData.age),
                weight: parseFloat(formData.weight),
            };

            const response = await animalApi.createAnimal(dataToSend);

            setSuccess(`Животное "${response.data.name}" успешно добавлено!`);
            setFormData({
                name: '',
                animal_type: 'dog',
                breed: '',
                age: '',
                weight: '',
                description: '',
                status: 'available',
                arrival_date: new Date().toISOString().split('T')[0],
                is_vaccinated: false,
                is_sterilized: false,
            });

            // Вызываем callback, если он есть
            if (onAnimalAdded) {
                onAnimalAdded(response.data);
            }
        } catch (err) {
            console.error('Error adding animal:', err);
            if (err.response && err.response.data && err.response.data.detail) {
                setError(`Ошибка: ${err.response.data.detail}`);
            } else {
                setError('Произошла ошибка при добавлении животного');
            }
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ maxWidth: '600px', margin: '0 auto' }}>
            <h2>Добавить новое животное</h2>

            {error && (
                <div style={{
                    padding: '10px',
                    backgroundColor: '#ffebee',
                    color: '#c62828',
                    marginBottom: '15px',
                    borderRadius: '4px',
                }}>
                    {error}
                </div>
            )}

            {success && (
                <div style={{
                    padding: '10px',
                    backgroundColor: '#e8f5e9',
                    color: '#2e7d32',
                    marginBottom: '15px',
                    borderRadius: '4px',
                }}>
                    {success}
                </div>
            )}

            <form onSubmit={handleSubmit} style={{
                backgroundColor: '#f5f5f5',
                padding: '20px',
                borderRadius: '8px',
            }}>
                {/* Имя */}
                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                        Имя животного *
                    </label>
                    <input
                        type="text"
                        name="name"
                        value={formData.name}
                        onChange={handleChange}
                        style={{
                            width: '100%',
                            padding: '8px',
                            border: '1px solid #ddd',
                            borderRadius: '4px',
                        }}
                        required
                    />
                </div>

                {/* Вид животного */}
                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                        Вид животного *
                    </label>
                    <select
                        name="animal_type"
                        value={formData.animal_type}
                        onChange={handleChange}
                        style={{
                            width: '100%',
                            padding: '8px',
                            border: '1px solid #ddd',
                            borderRadius: '4px',
                        }}
                    >
                        {animalTypes.map(type => (
                            <option key={type.value} value={type.value}>
                                {type.label}
                            </option>
                        ))}
                    </select>
                </div>

                {/* Порода */}
                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                        Порода
                    </label>
                    <input
                        type="text"
                        name="breed"
                        value={formData.breed}
                        onChange={handleChange}
                        style={{
                            width: '100%',
                            padding: '8px',
                            border: '1px solid #ddd',
                            borderRadius: '4px',
                        }}
                    />
                </div>

                {/* Возраст и вес в одной строке */}
                <div style={{ display: 'flex', gap: '15px', marginBottom: '15px' }}>
                    <div style={{ flex: 1 }}>
                        <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                            Возраст (месяцы) *
                        </label>
                        <input
                            type="number"
                            name="age"
                            value={formData.age}
                            onChange={handleChange}
                            min="0"
                            max="300"
                            style={{
                                width: '100%',
                                padding: '8px',
                                border: '1px solid #ddd',
                                borderRadius: '4px',
                            }}
                            required
                        />
                    </div>
                    <div style={{ flex: 1 }}>
                        <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                            Вес (кг) *
                        </label>
                        <input
                            type="number"
                            name="weight"
                            value={formData.weight}
                            onChange={handleChange}
                            min="0.1"
                            max="200"
                            step="0.1"
                            style={{
                                width: '100%',
                                padding: '8px',
                                border: '1px solid #ddd',
                                borderRadius: '4px',
                            }}
                            required
                        />
                    </div>
                </div>

                {/* Статус */}
                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                        Статус
                    </label>
                    <select
                        name="status"
                        value={formData.status}
                        onChange={handleChange}
                        style={{
                            width: '100%',
                            padding: '8px',
                            border: '1px solid #ddd',
                            borderRadius: '4px',
                        }}
                    >
                        {statusOptions.map(status => (
                            <option key={status.value} value={status.value}>
                                {status.label}
                            </option>
                        ))}
                    </select>
                </div>

                {/* Дата поступления */}
                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                        Дата поступления *
                    </label>
                    <input
                        type="date"
                        name="arrival_date"
                        value={formData.arrival_date}
                        onChange={handleChange}
                        style={{
                            width: '100%',
                            padding: '8px',
                            border: '1px solid #ddd',
                            borderRadius: '4px',
                        }}
                        required
                    />
                </div>

                {/* Описание */}
                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                        Описание
                    </label>
                    <textarea
                        name="description"
                        value={formData.description}
                        onChange={handleChange}
                        rows="3"
                        style={{
                            width: '100%',
                            padding: '8px',
                            border: '1px solid #ddd',
                            borderRadius: '4px',
                            resize: 'vertical',
                        }}
                    />
                </div>

                {/* Чекбоксы */}
                <div style={{ marginBottom: '20px' }}>
                    <label style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                        <input
                            type="checkbox"
                            name="is_vaccinated"
                            checked={formData.is_vaccinated}
                            onChange={handleChange}
                            style={{ marginRight: '8px' }}
                        />
                        Вакцинирован
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center' }}>
                        <input
                            type="checkbox"
                            name="is_sterilized"
                            checked={formData.is_sterilized}
                            onChange={handleChange}
                            style={{ marginRight: '8px' }}
                        />
                        Стерилизован
                    </label>
                </div>

                {/* Кнопка отправки */}
                <button
                    type="submit"
                    disabled={loading}
                    style={{
                        width: '100%',
                        padding: '12px',
                        backgroundColor: loading ? '#ccc' : '#4CAF50',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        fontSize: '16px',
                        cursor: loading ? 'not-allowed' : 'pointer',
                    }}
                >
                    {loading ? 'Добавление...' : 'Добавить животное'}
                </button>
            </form>
        </div>
    );
};

export default AddAnimalForm;