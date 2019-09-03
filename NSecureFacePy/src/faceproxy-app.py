import asyncio
from aiohttp import web


async def handler(request):
    print(request.content)
    return web.Response(text="RECEIVED")


async def main():
    server = web.Server(handler=handler)
    runner = web.ServerRunner(server)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 5001)
    await site.start()

    await asyncio.sleep(100*3600)

loop = asyncio.get_event_loop()

try:
    loop.run_until_complete(main())
except KeyboardInterrupt:
    pass

loop.close()
