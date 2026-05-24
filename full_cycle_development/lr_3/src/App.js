import React from "react";

import TodoForm from "./components/TodoForm";
import TodoList from "./components/TodoList";
import FilterButtons from "./components/FilterButtons";

function App() {

    return (
        <div style={{ padding: "20px" }}>

            <h1>To-Do List Redux</h1>

            <TodoForm />

            <FilterButtons />

            <TodoList />

        </div>
    );
}

export default App;