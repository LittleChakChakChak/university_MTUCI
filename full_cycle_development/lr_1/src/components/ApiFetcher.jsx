import { useState, useEffect } from 'react';

function ApiFetcher() {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('https://jsonplaceholder.typicode.com/posts')
      .then(response => response.json())
      .then(data => {
        setPosts(data.slice(0, 10));
        setLoading(false);
      })
      .catch(error => console.error('Ошибка:', error));
  }, []);

  return (
    <div style={{ border: '1px solid #ccc', padding: '20px', margin: '10px', borderRadius: '8px' }}>
      <h2>🌐 Задание 4: Данные с API</h2>
      {loading ? (
        <p>Загрузка...</p>
      ) : (
        <ul style={{ maxHeight: '300px', overflow: 'auto' }}>
          {posts.map(post => (
            <li key={post.id} style={{ marginBottom: '10px' }}>
              <strong>{post.title}</strong>
              <p style={{ margin: '5px 0 0 0', fontSize: '14px', color: '#555' }}>{post.body}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default ApiFetcher;