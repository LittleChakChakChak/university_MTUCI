import { useState, useEffect } from 'react';

function Timer() {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setSeconds(prev => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ border: '1px solid #ccc', padding: '20px', margin: '10px', borderRadius: '8px' }}>
      <h2>⏱️ Задание 3: Секундомер</h2>
      <p>Прошло секунд: <strong style={{ fontSize: '24px' }}>{seconds}</strong></p>
    </div>
  );
}

export default Timer;