import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';

// Импортируем компоненты
import AnimalList from './components/AnimalList';
import AddAnimalPage from './pages/AddAnimalPage';
import StatsPage from './pages/StatsPage';

function Home() {
    return (
        <div>
            <h2>Главная страница</h2>
            <p>Добро пожаловать в систему управления приютом для животных!</p>
            <div style={{ marginTop: '20px' }}>
                <h3>Быстрые действия:</h3>
                <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                    <Link to="/animals">
                        <button style={{
                            padding: '10px 20px',
                            backgroundColor: '#2196F3',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: 'pointer',
                        }}>
                            Просмотр животных
                        </button>
                    </Link>
                    <Link to="/add-animal">
                        <button style={{
                            padding: '10px 20px',
                            backgroundColor: '#4CAF50',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: 'pointer',
                        }}>
                            Добавить животное
                        </button>
                    </Link>
                </div>
            </div>
        </div>
    );
}

function Stats() {
    return (
        <div>
            <h2>Статистика приюта</h2>
            <p>Статистика будет отображаться здесь</p>
        </div>
    );
}

function App() {
    return (
        <Router>
            <div className="App" style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
                {/* Навигация */}
                <header style={{
                    backgroundColor: '#333',
                    color: 'white',
                    padding: '15px',
                    borderRadius: '8px',
                    marginBottom: '20px',
                }}>
                    <h1 style={{ margin: '0 0 10px 0' }}>🐾 Приют для животных</h1>
                    <nav>
                        <Link to="/" style={{ color: 'white', marginRight: '15px', textDecoration: 'none' }}>
                            Главная
                        </Link>
                        <Link to="/animals" style={{ color: 'white', marginRight: '15px', textDecoration: 'none' }}>
                            Все животные
                        </Link>
                        <Link to="/add-animal" style={{ color: 'white', marginRight: '15px', textDecoration: 'none' }}>
                            Добавить животное
                        </Link>
                        <Link to="/stats" style={{ color: 'white', textDecoration: 'none' }}>
                            Статистика
                        </Link>
                    </nav>
                </header>

                {/* Основной контент */}
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/animals" element={<AnimalList />} />
                    <Route path="/add-animal" element={<AddAnimalPage />} />
                    <Route path="/stats" element={<StatsPage />} />
                </Routes>

                {/* Футер */}
                <footer style={{
                    marginTop: '40px',
                    padding: '20px',
                    backgroundColor: '#f5f5f5',
                    borderRadius: '8px',
                    textAlign: 'center',
                }}>
                    <p>Система управления приютом для животных © 2025</p>
                    <p>Backend API: http://localhost:8000</p>
                </footer>
            </div>
        </Router>
    );
}

export default App;