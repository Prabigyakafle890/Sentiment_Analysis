from app import app

def handler(request, response):
    return app(request, response)