import React, { useEffect } from 'react';

const Notification = ({ message, type, onClose }) => {
    useEffect(() => {
        const timer = setTimeout(() => {
            onClose();
        }, 3000);

        return () => clearTimeout(timer);
    }, [onClose]);

    const styles = {
        success: {
            backgroundColor: '#d4edda',
            color: '#155724',
            borderColor: '#c3e6cb',
        },
        error: {
            backgroundColor: '#f8d7da',
            color: '#721c24',
            borderColor: '#f5c6cb',
        },
        info: {
            backgroundColor: '#d1ecf1',
            color: '#0c5460',
            borderColor: '#bee5eb',
        }
    };

    return (
        <div style={{
            position: 'fixed',
            top: '20px',
            right: '20px',
            zIndex: 1000,
            padding: '15px 20px',
            borderRadius: '8px',
            border: '1px solid',
            boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            minWidth: '300px',
            ...styles[type]
        }}>
            <span style={{ fontSize: '20px' }}>
                {type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}
            </span>
            <span>{message}</span>
            <button
                onClick={onClose}
                style={{
                    marginLeft: 'auto',
                    background: 'none',
                    border: 'none',
                    fontSize: '18px',
                    cursor: 'pointer',
                    color: 'inherit',
                }}
            >
                ×
            </button>
        </div>
    );
};

export default Notification;