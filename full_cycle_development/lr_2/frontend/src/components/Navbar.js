import React from 'react';
import { Link } from 'react-router-dom';
import {
    AppBar,
    Toolbar,
    Typography,
    Button,
    Container,
} from '@mui/material';
import PetsIcon from '@mui/icons-material/Pets';
import BarChartIcon from '@mui/icons-material/BarChart';

const Navbar = () => {
    return (
        <AppBar position="static">
            <Container maxWidth="lg">
                <Toolbar>
                    <PetsIcon sx={{ mr: 2 }} />
                    <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                        Приют для животных
                    </Typography>
                    <Button color="inherit" component={Link} to="/">
                        Главная
                    </Button>
                    <Button color="inherit" component={Link} to="/add">
                        Добавить животное
                    </Button>
                    <Button 
                        color="inherit" 
                        component={Link} 
                        to="/stats"
                        startIcon={<BarChartIcon />}
                    >
                        Статистика
                    </Button>
                </Toolbar>
            </Container>
        </AppBar>
    );
};

export default Navbar;
