import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Проверка, что главная страница грузится успешно."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'CAPTCHA' in response.data or b'captcha' in response.data.lower()