const express = require('express');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const cors = require('cors');
require('dotenv').config();

const db = require('./database');

const app = express();

app.use(express.json());
app.use(cors());

const SECRET_KEY = process.env.SECRET_KEY;


// Регистрация
app.post('/register', async (req, res) => {

    const { email, password } = req.body;

    if (!email || !password) {
        return res.status(400).json({
            message: 'Введите email и пароль'
        });
    }

    const hashedPassword = await bcrypt.hash(password, 10);

    db.run(
        `INSERT INTO users(email, password) VALUES (?, ?)`,
        [email, hashedPassword],
        function(err) {

            if (err) {
                return res.status(400).json({
                    message: 'Пользователь уже существует'
                });
            }

            res.json({
                message: 'Регистрация успешна'
            });
        }
    );
});


// Авторизация
app.post('/login', (req, res) => {

    const { email, password } = req.body;

    db.get(
        `SELECT * FROM users WHERE email = ?`,
        [email],
        async (err, user) => {

            if (!user) {
                return res.status(401).json({
                    message: 'Неверный email'
                });
            }

            const isValid = await bcrypt.compare(
                password,
                user.password
            );

            if (!isValid) {
                return res.status(401).json({
                    message: 'Неверный пароль'
                });
            }

            const token = jwt.sign(
                {
                    id: user.id,
                    email: user.email
                },
                SECRET_KEY,
                {
                    expiresIn: '1h'
                }
            );

            res.json({
                message: 'Успешный вход',
                token
            });
        }
    );
});


// Middleware проверки токена
function authenticateToken(req, res, next) {

    const authHeader = req.headers['authorization'];

    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
        return res.status(401).json({
            message: 'Токен отсутствует'
        });
    }

    jwt.verify(token, SECRET_KEY, (err, user) => {

        if (err) {
            return res.status(403).json({
                message: 'Неверный токен'
            });
        }

        req.user = user;

        next();
    });
}


// Защищенный маршрут
app.get('/profile', authenticateToken, (req, res) => {

    res.json({
        message: 'Доступ разрешен',
        user: req.user
    });
});


app.listen(process.env.PORT, () => {
    console.log(`Сервер запущен на порту ${process.env.PORT}`);
});