import React from "react";
import { useDispatch } from "react-redux";

import {
    setFilter,
    clearCompleted
} from "../redux/actions";

function FilterButtons() {

    const dispatch = useDispatch();

    return (
        <div>

            <button onClick={() => dispatch(setFilter("ALL"))}>
                Все
            </button>

            <button onClick={() => dispatch(setFilter("COMPLETED"))}>
                Выполненные
            </button>

            <button onClick={() => dispatch(setFilter("ACTIVE"))}>
                Невыполненные
            </button>

            <button onClick={() => dispatch(clearCompleted())}>
                Очистить выполненные
            </button>

        </div>
    );
}

export default FilterButtons;