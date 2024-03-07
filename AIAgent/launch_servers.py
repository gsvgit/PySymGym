import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from queue import Empty, Queue

import psutil
from aiohttp import web
from common.constants import (
    SERVER_WORKING_DIR,
    ResultsHandlerLinks,
    WebsocketSourceLinks,
)
from config import BrokerConfig, FeatureConfig, GeneralConfig
from connection.broker_conn.classes import (
    RunningServerInstanceInfo,
    ServerInstanceInfo,
    StaleServerInstanceInfo,
    Undefined,
    WSUrl,
)

routes = web.RouteTableDef()
logging.basicConfig(
    level=GeneralConfig.LOGGER_LEVEL,
    filename="instance_manager.log",
    filemode="w",
    format="%(asctime)s - p%(process)d: %(name)s - [%(levelname)s]: %(message)s",
)

ADDRES_IN_USE_ERROR = "System.Net.Sockets.SocketException (48): Address already in use"
avoid_same_free_port_lock = asyncio.Lock()


def next_free_port(port=35000, max_port=36000):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(("", port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise IOError("no free ports")


@routes.get(f"/{WebsocketSourceLinks.GET_WS}")
async def dequeue_instance(request):
    try:
        server_info = SERVER_INSTANCES.get(block=False)
        assert server_info.pid is Undefined
        server_info = await run_server_instance(
            port=server_info.port if DEBUG else None,
            start_server=FeatureConfig.ON_GAME_SERVER_RESTART.enabled,
        )
        logging.info(f"issued {server_info}: {psutil.Process(server_info.pid)}")
        return web.json_response(server_info.to_json())
    except Empty:
        logging.error("Couldn't dequeue instance, the queue is not replenishing")
        raise


@routes.post(f"/{WebsocketSourceLinks.POST_WS}")
async def enqueue_instance(request):
    returned_instance_info_raw = await request.read()
    returned_instance_info = ServerInstanceInfo.from_json(
        returned_instance_info_raw.decode("utf-8")
    )
    logging.info(f"got {returned_instance_info} from client")

    if FeatureConfig.ON_GAME_SERVER_RESTART.enabled:
        kill_server(returned_instance_info)
        returned_instance_info = ServerInstanceInfo(
            returned_instance_info.port, returned_instance_info.ws_url, pid=Undefined
        )

    SERVER_INSTANCES.put(returned_instance_info)
    logging.info(f"enqueue {returned_instance_info}")
    return web.HTTPOk()


@routes.post(f"/{ResultsHandlerLinks.POST_RES}")
async def append_results(request):
    global RESULTS
    data = await request.read()
    decoded = data.decode("utf-8")
    RESULTS.append(decoded)
    return web.HTTPOk()


@routes.get(f"/{ResultsHandlerLinks.GET_RES}")
async def send_and_clear_results(request):
    global RESULTS
    if not RESULTS:
        raise RuntimeError("Must play a game first")
    rst = json.dumps(RESULTS)
    RESULTS = []
    return web.Response(text=rst)


def get_socket_url(port: int) -> WSUrl:
    return f"ws://{BrokerConfig.BROKER_PORT}:{port}/gameServer"


async def run_server_instance(*, should_start_server: bool) -> ServerInstanceInfo:
    launch_server = [
        "dotnet",
        "VSharp.ML.GameServer.Runner.dll",
        "--mode",
        "server",
        "--port",
    ]
    if not should_start_server:
        return StaleServerInstanceInfo(None, "Empty WSUrl", pid=Undefined)

    def start_server() -> tuple[subprocess.Popen[bytes], int]:
        port = next_free_port()
        return (
            subprocess.Popen(
                launch_server + [str(port)],
                stdout=subprocess.PIPE,
                start_new_session=True,
                cwd=SERVER_WORKING_DIR,
            ),
            port,
        )

    async with avoid_same_free_port_lock:
        proc, port = start_server()

        proc_out = proc.stdout.read()
        while ADDRES_IN_USE_ERROR in proc_out:
            logging.warning(f"{port=} was already in use, trying new port...")
            proc, port = start_server()

        print(proc_out.decode("utf-8"), end="")

    PROCS.append(proc.pid)
    logging.info(
        f"running new instance on {port=} with {proc.pid=}:"
        + f"{proc.pid}: "
        + " ".join(launch_server + [str(port)])
    )

    ws_url = get_socket_url(port)
    return RunningServerInstanceInfo(port, ws_url, proc.pid)


async def run_servers(size: int) -> list[ServerInstanceInfo]:
    servers_start_tasks = []

    async def run():
        server_info = await run_server_instance(should_start_server=False)
        servers_start_tasks.append(server_info)

    asyncio.gather(run() for _ in range(size))

    return servers_start_tasks


def kill_server(server_instance: ServerInstanceInfo):
    os.kill(server_instance.pid, signal.SIGKILL)
    PROCS.remove(server_instance.pid)

    proc_info = psutil.Process(server_instance.pid)
    wait_for_reset_retries = FeatureConfig.ON_GAME_SERVER_RESTART.wait_for_reset_retries

    while wait_for_reset_retries:
        logging.info(
            f"Waiting for {server_instance} to die, {wait_for_reset_retries} retries left"
        )
        if proc_info.status() in (psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE):
            logging.info(f"killed {proc_info}")
            return
        time.sleep(FeatureConfig.ON_GAME_SERVER_RESTART.wait_for_reset_time)
        wait_for_reset_retries -= 1

    raise RuntimeError(f"{server_instance} could not be killed")


def kill_process(pid: int):
    os.kill(pid, signal.SIGKILL)
    PROCS.remove(pid)


@contextmanager
def server_manager(server_queue: Queue[ServerInstanceInfo], *, size: int, debug: bool):
    global PROCS

    servers_info = asyncio.run(run_servers(size=size))

    for server_info in servers_info:
        server_queue.put(server_info)
    try:
        yield
    finally:
        for proc in list(PROCS):
            kill_process(proc)
        PROCS = []


def main():
    global SERVER_INSTANCES, PROCS, RESULTS, DEBUG
    parser = argparse.ArgumentParser(description="V# instances launcher")
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        help="dont launch servers if set",
    )

    args = parser.parse_args()
    DEBUG = args.debug or False
    # restart should be disabled for debug mode
    if DEBUG and FeatureConfig.ON_GAME_SERVER_RESTART.enabled:
        raise RuntimeError("Disable ON_GAME_SERVER_RESTART feature to use debug mode")

    # Queue[ServerInstanceInfo]
    SERVER_INSTANCES = Queue()
    PROCS = []
    RESULTS = []

    with server_manager(SERVER_INSTANCES, size=GeneralConfig.SERVER_COUNT, debug=DEBUG):
        app = web.Application()
        app.add_routes(routes)
        web.run_app(app, port=BrokerConfig.BROKER_PORT)


if __name__ == "__main__":
    main()
