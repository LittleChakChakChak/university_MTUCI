import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div style={{ border: '1px solid #ccc', padding: '20px', margin: '10px', borderRadius: '8px' }}>
      <h2>📊 Задание 1: Счётчик</h2>
      <p>Значение: <strong style={{ fontSize: '24px' }}>{count}</strong></p>
      <button onClick={() => setCount(count + 1)} style={{ marginRight: '10px' }}>+1</button>
      <button onClick={() => setCount(count - 1)}>-1</button>
    </div>
  );
}

export default Counter;