import React from "react";

import { useDispatch } from "react-redux";

import {
    deleteTodo,
    toggleTodo
} from "../redux/actions";

function TodoItem({ todo }) {

    const dispatch = useDispatch();

    return (
        <li>

            <p>
                <strong>ID:</strong> {todo.id}
            </p>

            <p>
                <strong>Задача:</strong> {todo.text}
            </p>

            <p>
                <strong>Статус:</strong>

                {
                    todo.completed
                        ? " Выполнена"
                        : " Не выполнена"
                }
            </p>

            <button
                onClick={() =>
                    dispatch(toggleTodo(todo.id))
                }
            >
                Изменить статус
            </button>

            <button
                onClick={() =>
                    dispatch(deleteTodo(todo.id))
                }
            >
                Удалить
            </button>

            <hr />

        </li>
    );
}

export default TodoItem;