import Counter from './components/Counter';
import TextInput from './components/TextInput';
import Timer from './components/Timer';
import ApiFetcher from './components/ApiFetcher';

function App() {
  return (
    <div style={{ fontFamily: 'Arial', padding: '20px' }}>
      <h1>Лабораторная работа №1: Базовая работа с хуками в React</h1>
      <Counter />
      <TextInput />
      <Timer />
      <ApiFetcher />
    </div>
  );
}

export default App;