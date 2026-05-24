import React from "react";
import { useSelector } from "react-redux";

import TodoItem from "./TodoItem";

function TodoList() {

    const todos = useSelector(state => state.todos);
    const filter = useSelector(state => state.filter);

    const filteredTodos = todos.filter(todo => {

        if (filter === "COMPLETED") {
            return todo.completed;
        }

        if (filter === "ACTIVE") {
            return !todo.completed;
        }

        return true;
    });

    return (
        <div>

            <h3>
                Количество задач: {filteredTodos.length}
            </h3>

            <ul>
                {
                    filteredTodos.map(todo => (
                        <TodoItem
                            key={todo.id}
                            todo={todo}
                        />
                    ))
                }
            </ul>

        </div>
    );
}

export default TodoList;