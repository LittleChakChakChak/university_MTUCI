import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const animalApi = {
    // Получить всех животных
    getAllAnimals: (params) => api.get('/animals/', { params }),

    // Получить животное по ID
    getAnimalById: (id) => api.get(`/animals/${id}`),

    // Создать новое животное
    createAnimal: (animalData) => api.post('/animals/', animalData),

    // Обновить животное
    updateAnimal: (id, animalData) => api.put(`/animals/${id}`, animalData),

    // Удалить животное
    deleteAnimal: (id) => api.delete(`/animals/${id}`),

    // УСЫНОВИТЬ животное - исправлено на GET
    adoptAnimal: (id) => api.get(`/animals/${id}/adopt`),

    // Перевести на мед уход
    moveToMedical: (id) => api.get(`/animals/${id}/medical`),

    // Получить статистику
    getStats: () => api.get('/stats/'),

    // Получить количество по типу
    getTypeCount: (type) => api.get(`/animals/types/${type}/count`),
};

// Добавим обработчик ошибок
api.interceptors.response.use(
    response => response,
    error => {
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
    }
);