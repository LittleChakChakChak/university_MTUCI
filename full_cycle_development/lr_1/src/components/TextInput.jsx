import { useState } from 'react';

function TextInput() {
  const [text, setText] = useState('');

  return (
    <div style={{ border: '1px solid #ccc', padding: '20px', margin: '10px', borderRadius: '8px' }}>
      <h2>✏️ Задание 2: Отслеживание ввода</h2>
      <input
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Введите текст..."
        style={{ padding: '8px', width: '200px' }}
      />
      <p>Вы ввели: <strong>{text || '...'}</strong></p>
    </div>
  );
}

export default TextInput;