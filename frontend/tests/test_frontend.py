class TestFrontend:
    def test_frontend_exists(self):
        from frontend import app
        assert app is not None
