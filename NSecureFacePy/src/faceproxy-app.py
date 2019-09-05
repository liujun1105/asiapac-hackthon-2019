import asyncio
from aiohttp import web


async def handler(request):
    print(request.content)
    return web.Response(text="RECEIVED")


app = web.Application()
app.add_routes([web.get('/', handler)])

web.run_app(app, host='0.0.0.0', port=5001)
