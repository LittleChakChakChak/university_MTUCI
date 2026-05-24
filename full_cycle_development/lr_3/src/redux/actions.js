export const ADD_TODO = "ADD_TODO";
export const DELETE_TODO = "DELETE_TODO";
export const TOGGLE_TODO = "TOGGLE_TODO";
export const CLEAR_COMPLETED = "CLEAR_COMPLETED";
export const SET_FILTER = "SET_FILTER";

export const addTodo = (text) => ({
    type: ADD_TODO,
    payload: text
});

export const deleteTodo = (id) => ({
    type: DELETE_TODO,
    payload: id
});

export const toggleTodo = (id) => ({
    type: TOGGLE_TODO,
    payload: id
});

export const clearCompleted = () => ({
    type: CLEAR_COMPLETED
});

export const setFilter = (filter) => ({
    type: SET_FILTER,
    payload: filter
});