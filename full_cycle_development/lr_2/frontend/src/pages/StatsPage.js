import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { animalApi } from '../services/api';
import {
    PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
    ResponsiveContainer, LineChart, Line
} from 'recharts';

const StatsPage = () => {
    const [stats, setStats] = useState(null);
    const [animals, setAnimals] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [timeRange, setTimeRange] = useState('all'); // all, month, year

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        try {
            setLoading(true);

            // Загружаем статистику
            const statsResponse = await animalApi.getStats();
            setStats(statsResponse.data);

            // Загружаем всех животных для дополнительной статистики
            const animalsResponse = await animalApi.getAllAnimals();
            setAnimals(animalsResponse.data);

            setError('');
        } catch (err) {
            console.error('Error loading stats:', err);
            setError('Не удалось загрузить статистику');
        } finally {
            setLoading(false);
        }
    };

    // Подготовка данных для круговой диаграммы по типам
    const prepareTypeData = () => {
        if (!stats?.by_type) return [];

        const typeLabels = {
            dog: 'Собаки',
            cat: 'Кошки',
            bird: 'Птицы',
            rabbit: 'Кролики',
            other: 'Другие'
        };

        const colors = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

        return Object.entries(stats.by_type).map(([type, count], index) => ({
            name: typeLabels[type] || type,
            value: count,
            color: colors[index % colors.length]
        }));
    };

    // Подготовка данных для столбчатой диаграммы по статусам
    const prepareStatusData = () => {
        const statusData = {
            available: { name: 'Доступны', count: 0, color: '#4CAF50' },
            adopted: { name: 'Усыновлены', count: 0, color: '#2196F3' },
            medical_care: { name: 'Мед. уход', count: 0, color: '#FF9800' },
            foster_care: { name: 'Передержка', count: 0, color: '#9C27B0' },
            quarantine: { name: 'Карантин', count: 0, color: '#F44336' }
        };

        animals.forEach(animal => {
            if (statusData[animal.status]) {
                statusData[animal.status].count++;
            }
        });

        return Object.values(statusData);
    };

    // Подготовка данных для графика поступления по месяцам
    const prepareMonthlyData = () => {
        const monthlyData = {};

        animals.forEach(animal => {
            const date = new Date(animal.arrival_date);
            const monthYear = `${date.getFullYear()}-${date.getMonth() + 1}`;

            if (!monthlyData[monthYear]) {
                monthlyData[monthYear] = {
                    month: monthYear,
                    count: 0,
                    adopted: 0,
                    available: 0
                };
            }

            monthlyData[monthYear].count++;

            if (animal.status === 'adopted') {
                monthlyData[monthYear].adopted++;
            } else if (animal.status === 'available') {
                monthlyData[monthYear].available++;
            }
        });

        return Object.values(monthlyData)
            .sort((a, b) => a.month.localeCompare(b.month))
            .slice(-6); // Последние 6 месяцев
    };

    // Расчет средней длительности пребывания
    const calculateAverageStay = () => {
        const adoptedAnimals = animals.filter(a => a.status === 'adopted' && a.adoption_date);

        if (adoptedAnimals.length === 0) return 0;

        const totalDays = adoptedAnimals.reduce((sum, animal) => {
            const arrival = new Date(animal.arrival_date);
            const adoption = new Date(animal.adoption_date);
            const days = Math.floor((adoption - arrival) / (1000 * 60 * 60 * 24));
            return sum + days;
        }, 0);

        return Math.round(totalDays / adoptedAnimals.length);
    };

    // Статистика по возрасту
    const prepareAgeData = () => {
        const ageGroups = {
            '0-6': { name: '0-6 мес', count: 0 },
            '7-12': { name: '7-12 мес', count: 0 },
            '13-24': { name: '1-2 года', count: 0 },
            '25-60': { name: '2-5 лет', count: 0 },
            '60+': { name: '5+ лет', count: 0 }
        };

        animals.forEach(animal => {
            const age = animal.age;
            if (age <= 6) ageGroups['0-6'].count++;
            else if (age <= 12) ageGroups['7-12'].count++;
            else if (age <= 24) ageGroups['13-24'].count++;
            else if (age <= 60) ageGroups['25-60'].count++;
            else ageGroups['60+'].count++;
        });

        return Object.values(ageGroups);
    };

    if (loading) {
        return (
            <div style={{ textAlign: 'center', padding: '40px' }}>
                <div style={{ fontSize: '24px', marginBottom: '20px' }}>📊</div>
                <p>Загрузка статистики...</p>
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
                    onClick={loadData}
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

    const typeData = prepareTypeData();
    const statusData = prepareStatusData();
    const monthlyData = prepareMonthlyData();
    const ageData = prepareAgeData();
    const avgStay = calculateAverageStay();

    return (
        <div style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>
            {/* Хлебные крошки */}
            <div style={{ marginBottom: '20px' }}>
                <Link to="/" style={{ color: '#2196F3', textDecoration: 'none', marginRight: '10px' }}>
                    Главная
                </Link>
                <span style={{ color: '#6c757d' }}>›</span>
                <span style={{ marginLeft: '10px', color: '#495057', fontWeight: '500' }}>
                    Статистика приюта
                </span>
            </div>

            {/* Заголовок и кнопка обновления */}
            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '30px',
            }}>
                <h1 style={{ margin: 0, color: '#343a40' }}>
                    📊 Статистика приюта для животных
                </h1>
                <div style={{ display: 'flex', gap: '10px' }}>
                    <select
                        value={timeRange}
                        onChange={(e) => setTimeRange(e.target.value)}
                        style={{
                            padding: '8px 16px',
                            border: '1px solid #ced4da',
                            borderRadius: '6px',
                            backgroundColor: 'white',
                            fontSize: '14px',
                        }}
                    >
                        <option value="all">За всё время</option>
                        <option value="month">За месяц</option>
                        <option value="year">За год</option>
                    </select>
                    <button
                        onClick={loadData}
                        style={{
                            padding: '8px 20px',
                            backgroundColor: '#28a745',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            fontSize: '14px',
                            fontWeight: '500',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                        }}
                    >
                        🔄 Обновить
                    </button>
                </div>
            </div>

            {/* Основные показатели */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
                gap: '20px',
                marginBottom: '30px',
            }}>
                {/* Карточка: Всего животных */}
                <div style={{
                    backgroundColor: '#e3f2fd',
                    padding: '25px',
                    borderRadius: '12px',
                    borderLeft: '6px solid #2196F3',
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', marginBottom: '15px' }}>
                        <div style={{
                            backgroundColor: '#bbdefb',
                            padding: '12px',
                            borderRadius: '10px',
                            marginRight: '15px',
                        }}>
                            <span style={{ fontSize: '24px' }}>🐾</span>
                        </div>
                        <div>
                            <h3 style={{ margin: 0, color: '#1976d2' }}>Всего животных</h3>
                            <p style={{ margin: '5px 0 0 0', color: '#666', fontSize: '14px' }}>
                                В приюте
                            </p>
                        </div>
                    </div>
                    <div style={{ fontSize: '36px', fontWeight: 'bold', color: '#1565c0' }}>
                        {stats?.total_animals || 0}
                    </div>
                </div>

                {/* Карточка: Доступны для усыновления */}
                <div style={{
                    backgroundColor: '#e8f5e9',
                    padding: '25px',
                    borderRadius: '12px',
                    borderLeft: '6px solid #4CAF50',
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', marginBottom: '15px' }}>
                        <div style={{
                            backgroundColor: '#c8e6c9',
                            padding: '12px',
                            borderRadius: '10px',
                            marginRight: '15px',
                        }}>
                            <span style={{ fontSize: '24px' }}>🏡</span>
                        </div>
                        <div>
                            <h3 style={{ margin: 0, color: '#388e3c' }}>Доступны</h3>
                            <p style={{ margin: '5px 0 0 0', color: '#666', fontSize: '14px' }}>
                                Для усыновления
                            </p>
                        </div>
                    </div>
                    <div style={{ fontSize: '36px', fontWeight: 'bold', color: '#2e7d32' }}>
                        {stats?.available_count || 0}
                    </div>
                </div>

                {/* Карточка: Усыновлены */}
                <div style={{
                    backgroundColor: '#e1f5fe',
                    padding: '25px',
                    borderRadius: '12px',
                    borderLeft: '6px solid #03a9f4',
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', marginBottom: '15px' }}>
                        <div style={{
                            backgroundColor: '#b3e5fc',
                            padding: '12px',
                            borderRadius: '10px',
                            marginRight: '15px',
                        }}>
                            <span style={{ fontSize: '24px' }}>❤️</span>
                        </div>
                        <div>
                            <h3 style={{ margin: 0, color: '#0277bd' }}>Усыновлены</h3>
                            <p style={{ margin: '5px 0 0 0', color: '#666', fontSize: '14px' }}>
                                Нашли дом
                            </p>
                        </div>
                    </div>
                    <div style={{ fontSize: '36px', fontWeight: 'bold', color: '#01579b' }}>
                        {stats?.adopted_count || 0}
                    </div>
                    <div style={{ fontSize: '14px', color: '#666', marginTop: '5px' }}>
                        {stats?.adopted_count > 0
                            ? `${Math.round((stats.adopted_count / stats.total_animals) * 100)}% от всех`
                            : 'Пока нет усыновлений'}
                    </div>
                </div>

                {/* Карточка: Средний возраст */}
                <div style={{
                    backgroundColor: '#fff3e0',
                    padding: '25px',
                    borderRadius: '12px',
                    borderLeft: '6px solid #ff9800',
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', marginBottom: '15px' }}>
                        <div style={{
                            backgroundColor: '#ffe0b2',
                            padding: '12px',
                            borderRadius: '10px',
                            marginRight: '15px',
                        }}>
                            <span style={{ fontSize: '24px' }}>🎂</span>
                        </div>
                        <div>
                            <h3 style={{ margin: 0, color: '#f57c00' }}>Средний возраст</h3>
                            <p style={{ margin: '5px 0 0 0', color: '#666', fontSize: '14px' }}>
                                Животных в приюте
                            </p>
                        </div>
                    </div>
                    <div style={{ fontSize: '36px', fontWeight: 'bold', color: '#e65100' }}>
                        {stats?.average_age || 0}
                        <span style={{ fontSize: '18px', marginLeft: '5px' }}>мес.</span>
                    </div>
                    <div style={{ fontSize: '14px', color: '#666', marginTop: '5px' }}>
                        (~{(stats?.average_age / 12).toFixed(1)} лет)
                    </div>
                </div>
            </div>

            {/* Графики - первая строка */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))',
                gap: '25px',
                marginBottom: '25px',
            }}>
                {/* Круговая диаграмма: Распределение по видам */}
                <div style={{
                    backgroundColor: 'white',
                    padding: '25px',
                    borderRadius: '12px',
                    boxShadow: '0 2px 10px rgba(0,0,0,0.08)',
                }}>
                    <h3 style={{ margin: '0 0 20px 0', color: '#343a40' }}>
                        🐕 Распределение по видам животных
                    </h3>
                    {typeData.length > 0 ? (
                        <div style={{ height: '300px' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie
                                        data={typeData}
                                        cx="50%"
                                        cy="50%"
                                        labelLine={false}
                                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                                        outerRadius={80}
                                        fill="#8884d8"
                                        dataKey="value"
                                    >
                                        {typeData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Pie>
                                    <Tooltip
                                        formatter={(value) => [`${value} животных`, 'Количество']}
                                    />
                                    <Legend />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>
                    ) : (
                        <div style={{
                            textAlign: 'center',
                            padding: '40px',
                            color: '#6c757d'
                        }}>
                            Нет данных для отображения
                        </div>
                    )}
                </div>

                {/* Столбчатая диаграмма: Статусы животных */}
                <div style={{
                    backgroundColor: 'white',
                    padding: '25px',
                    borderRadius: '12px',
                    boxShadow: '0 2px 10px rgba(0,0,0,0.08)',
                }}>
                    <h3 style={{ margin: '0 0 20px 0', color: '#343a40' }}>
                        📊 Статусы животных
                    </h3>
                    {statusData.length > 0 ? (
                        <div style={{ height: '300px' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart
                                    data={statusData}
                                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                                >
                                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                                    <XAxis
                                        dataKey="name"
                                        angle={-45}
                                        textAnchor="end"
                                        height={60}
                                    />
                                    <YAxis />
                                    <Tooltip
                                        formatter={(value) => [`${value} животных`, 'Количество']}
                                    />
                                    <Legend />
                                    <Bar
                                        dataKey="count"
                                        name="Количество животных"
                                        radius={[4, 4, 0, 0]}
                                    >
                                        {statusData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    ) : (
                        <div style={{
                            textAlign: 'center',
                            padding: '40px',
                            color: '#6c757d'
                        }}>
                            Нет данных для отображения
                        </div>
                    )}
                </div>
            </div>

            {/* Графики - вторая строка */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))',
                gap: '25px',
                marginBottom: '25px',
            }}>
                {/* Линейный график: Поступление по месяцам */}
                <div style={{
                    backgroundColor: 'white',
                    padding: '25px',
                    borderRadius: '12px',
                    boxShadow: '0 2px 10px rgba(0,0,0,0.08)',
                }}>
                    <h3 style={{ margin: '0 0 20px 0', color: '#343a40' }}>
                        📈 Поступление животных по месяцам
                    </h3>
                    {monthlyData.length > 0 ? (
                        <div style={{ height: '300px' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart
                                    data={monthlyData}
                                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                                >
                                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                                    <XAxis
                                        dataKey="month"
                                        tickFormatter={(value) => {
                                            const [year, month] = value.split('-');
                                            return `${month}/${year.slice(2)}`;
                                        }}
                                    />
                                    <YAxis />
                                    <Tooltip
                                        formatter={(value) => [`${value} животных`, 'Количество']}
                                        labelFormatter={(label) => {
                                            const [year, month] = label.split('-');
                                            return `Месяц: ${month}/${year}`;
                                        }}
                                    />
                                    <Legend />
                                    <Line
                                        type="monotone"
                                        dataKey="count"
                                        name="Поступило"
                                        stroke="#8884d8"
                                        strokeWidth={3}
                                        dot={{ r: 4 }}
                                        activeDot={{ r: 8 }}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="adopted"
                                        name="Усыновлено"
                                        stroke="#82ca9d"
                                        strokeWidth={2}
                                        strokeDasharray="5 5"
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    ) : (
                        <div style={{
                            textAlign: 'center',
                            padding: '40px',
                            color: '#6c757d'
                        }}>
                            Нет данных для отображения
                        </div>
                    )}
                </div>

                {/* Столбчатая диаграмма: Возрастные группы */}
                <div style={{
                    backgroundColor: 'white',
                    padding: '25px',
                    borderRadius: '12px',
                    boxShadow: '0 2px 10px rgba(0,0,0,0.08)',
                }}>
                    <h3 style={{ margin: '0 0 20px 0', color: '#343a40' }}>
                        👶 Возрастные группы животных
                    </h3>
                    {ageData.length > 0 ? (
                        <div style={{ height: '300px' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart
                                    data={ageData}
                                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                                >
                                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                                    <XAxis dataKey="name" />
                                    <YAxis />
                                    <Tooltip
                                        formatter={(value) => [`${value} животных`, 'Количество']}
                                    />
                                    <Legend />
                                    <Bar
                                        dataKey="count"
                                        name="Количество животных"
                                        fill="#ffc658"
                                        radius={[4, 4, 0, 0]}
                                    />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    ) : (
                        <div style={{
                            textAlign: 'center',
                            padding: '40px',
                            color: '#6c757d'
                        }}>
                            Нет данных для отображения
                        </div>
                    )}
                </div>
            </div>

            {/* Дополнительная статистика */}
            <div style={{
                backgroundColor: 'white',
                padding: '30px',
                borderRadius: '12px',
                boxShadow: '0 2px 10px rgba(0,0,0,0.08)',
                marginBottom: '30px',
            }}>
                <h3 style={{ margin: '0 0 25px 0', color: '#343a40' }}>
                    📋 Детальная статистика
                </h3>

                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                    gap: '20px',
                }}>
                    {/* Вакцинация */}
                    <div style={{
                        backgroundColor: '#f8f9fa',
                        padding: '20px',
                        borderRadius: '8px',
                        border: '1px solid #e9ecef',
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                            <span style={{ fontSize: '20px', marginRight: '10px' }}>💉</span>
                            <h4 style={{ margin: 0, color: '#495057' }}>Вакцинация</h4>
                        </div>
                        <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#28a745' }}>
                            {animals.filter(a => a.is_vaccinated).length}
                        </div>
                        <div style={{ fontSize: '14px', color: '#6c757d' }}>
                            вакцинированных животных
                        </div>
                        <div style={{ fontSize: '12px', color: '#868e96', marginTop: '5px' }}>
                            {animals.length > 0
                                ? `${Math.round((animals.filter(a => a.is_vaccinated).length / animals.length) * 100)}% от всех`
                                : 'Нет данных'}
                        </div>
                    </div>

                    {/* Стерилизация */}
                    <div style={{
                        backgroundColor: '#f8f9fa',
                        padding: '20px',
                        borderRadius: '8px',
                        border: '1px solid #e9ecef',
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                            <span style={{ fontSize: '20px', marginRight: '10px' }}>✂️</span>
                            <h4 style={{ margin: 0, color: '#495057' }}>Стерилизация</h4>
                        </div>
                        <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#dc3545' }}>
                            {animals.filter(a => a.is_sterilized).length}
                        </div>
                        <div style={{ fontSize: '14px', color: '#6c757d' }}>
                            стерилизованных животных
                        </div>
                        <div style={{ fontSize: '12px', color: '#868e96', marginTop: '5px' }}>
                            {animals.length > 0
                                ? `${Math.round((animals.filter(a => a.is_sterilized).length / animals.length) * 100)}% от всех`
                                : 'Нет данных'}
                        </div>
                    </div>

                    {/* Среднее время пребывания */}
                    <div style={{
                        backgroundColor: '#f8f9fa',
                        padding: '20px',
                        borderRadius: '8px',
                        border: '1px solid #e9ecef',
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                            <span style={{ fontSize: '20px', marginRight: '10px' }}>⏱️</span>
                            <h4 style={{ margin: 0, color: '#495057' }}>Среднее время</h4>
                        </div>
                        <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#17a2b8' }}>
                            {avgStay}
                        </div>
                        <div style={{ fontSize: '14px', color: '#6c757d' }}>
                            дней до усыновления
                        </div>
                        <div style={{ fontSize: '12px', color: '#868e96', marginTop: '5px' }}>
                            {animals.filter(a => a.status === 'adopted').length > 0
                                ? 'Среднее по усыновленным'
                                : 'Нет усыновленных животных'}
                        </div>
                    </div>

                    {/* Самый долгий житель */}
                    <div style={{
                        backgroundColor: '#f8f9fa',
                        padding: '20px',
                        borderRadius: '8px',
                        border: '1px solid #e9ecef',
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                            <span style={{ fontSize: '20px', marginRight: '10px' }}>👴</span>
                            <h4 style={{ margin: 0, color: '#495057' }}>Самый долгий</h4>
                        </div>
                        {animals.length > 0 ? (
                            <>
                                <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#6f42c1' }}>
                                    {animals.reduce((oldest, current) =>
                                        current.age > oldest.age ? current : oldest
                                    ).name}
                                </div>
                                <div style={{ fontSize: '14px', color: '#6c757d' }}>
                                    {Math.max(...animals.map(a => a.age))} мес.
                                </div>
                                <div style={{ fontSize: '12px', color: '#868e96', marginTop: '5px' }}>
                                    Самый взрослый в приюте
                                </div>
                            </>
                        ) : (
                            <div style={{ fontSize: '14px', color: '#6c757d' }}>
                                Нет данных
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Кнопки действий */}
            <div style={{
                display: 'flex',
                justifyContent: 'center',
                gap: '15px',
                marginTop: '30px',
            }}>
                <Link to="/animals">
                    <button style={{
                        padding: '12px 30px',
                        backgroundColor: '#6c757d',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontSize: '16px',
                        fontWeight: '500',
                    }}>
                        📋 К списку животных
                    </button>
                </Link>
                <Link to="/add-animal">
                    <button style={{
                        padding: '12px 30px',
                        backgroundColor: '#28a745',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontSize: '16px',
                        fontWeight: '500',
                    }}>
                        ➕ Добавить животное
                    </button>
                </Link>
                <button
                    onClick={loadData}
                    style={{
                        padding: '12px 30px',
                        backgroundColor: '#17a2b8',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontSize: '16px',
                        fontWeight: '500',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                    }}
                >
                    🔄 Обновить статистику
                </button>
            </div>

            {/* Футер статистики */}
            <div style={{
                marginTop: '30px',
                padding: '20px',
                backgroundColor: '#f8f9fa',
                borderRadius: '8px',
                fontSize: '14px',
                color: '#6c757d',
                textAlign: 'center',
            }}>
                <div style={{ marginBottom: '10px' }}>
                    <strong>Отчет сгенерирован:</strong> {new Date().toLocaleString('ru-RU')}
                </div>
                <div>
                    <span style={{ marginRight: '15px' }}>
                        🏠 Приют для животных
                    </span>
                    <span style={{ marginRight: '15px' }}>
                        📧 admin@shelter.ru
                    </span>
                    <span>
                        📞 +7 (XXX) XXX-XX-XX
                    </span>
                </div>
            </div>
        </div>
    );
};

export default StatsPage;