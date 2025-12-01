import React from 'react';
import { useNavigate } from 'react-router-dom';
import AddAnimalForm from '../components/AddAnimalForm';

const AddAnimalPage = () => {
    const navigate = useNavigate();

    const handleAnimalAdded = (animal) => {
        // Можно добавить перенаправление или другие действия
        console.log('Животное добавлено:', animal);
        // Через 2 секунды перенаправить на список животных
        setTimeout(() => {
            navigate('/animals');
        }, 2000);
    };

    return (
        <div style={{ padding: '20px' }}>
            <button
                onClick={() => navigate('/animals')}
                style={{
                    marginBottom: '20px',
                    padding: '8px 16px',
                    backgroundColor: '#2196F3',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                }}
            >
                ← Назад к списку животных
            </button>

            <AddAnimalForm onAnimalAdded={handleAnimalAdded} />
        </div>
    );
};

export default AddAnimalPage;