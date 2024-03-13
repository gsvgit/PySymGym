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

from common.constants import SERVER_WORKING_DIR
from config import BrokerConfig, FeatureConfig, GeneralConfig
from connection.broker_conn.classes import ServerInstanceInfo, Undefined, WSUrl

routes = web.RouteTableDef()
logging.basicConfig(
    level=GeneralConfig.LOGGER_LEVEL,
    filename="instance_manager.log",
    filemode="w",
    format="%(asctime)s - p%(process)d: %(name)s - [%(levelname)s]: %(message)s",
)

FAILED_TO_INSTANTIATE_ERROR = "TCP server failed"
avoid_same_free_port_lock = asyncio.Lock()


def next_free_port(min_port=35001, max_port=36000):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while min_port <= max_port:
        try:
            sock.bind(("", min_port))
            sock.close()
            return min_port
        except OSError:
            min_port += 1
    raise IOError("no free ports")


@routes.get("/get_ws")
async def dequeue_instance(request):
    try:
        server_info = SERVER_INSTANCES.get(block=False)
        assert server_info.pid is Undefined
        server_info = await run_server_instance(
            should_start_server=FeatureConfig.ON_GAME_SERVER_RESTART.enabled,
        )
        logging.info(f"issued {server_info}: {psutil.Process(server_info.pid)}")
        return web.json_response(server_info.to_json())
    except Empty:
        logging.error("Couldn't dequeue instance, the queue is not replenishing")
        raise


@routes.post("/post_ws")
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


@routes.post("/send_res")
async def append_results(request):
    global RESULTS
    data = await request.read()
    decoded = data.decode("utf-8")
    RESULTS.append(decoded)
    return web.HTTPOk()


@routes.get("/recv_res")
async def send_and_clear_results(request):
    global RESULTS
    if not RESULTS:
        raise RuntimeError("Must play a game first")
    rst = json.dumps(RESULTS)
    RESULTS = []
    return web.Response(text=rst)


def get_socket_url(port: int) -> WSUrl:
    return f"ws://0.0.0.0:{port}/gameServer"


async def run_server_instance(should_start_server: bool) -> ServerInstanceInfo:
    async with avoid_same_free_port_lock:
        launch_server = [
            "dotnet",
            "VSharp.ML.GameServer.Runner.dll",
            "--mode",
            "server",
            "--port",
        ]
        if not should_start_server:
            return ServerInstanceInfo(0, "None", pid=Undefined)

        def start_server() -> tuple[subprocess.Popen[bytes], int]:
            port = next_free_port()
            proc = subprocess.Popen(
                launch_server + [str(port)],
                stdout=subprocess.PIPE,
                start_new_session=True,
                cwd=SERVER_WORKING_DIR,
            )
            logging.info(f"bash exec cmd: {' '.join(launch_server + [str(port)])}")
            _ = proc.stdout.readline()
            proc_out = proc.stdout.readline().decode("utf-8").strip("\n")

            return proc, port, proc_out

    async with avoid_same_free_port_lock:
        proc, port, proc_out = start_server()

        while FAILED_TO_INSTANTIATE_ERROR in proc_out:
            logging.warning(
                f"{port=} was already in use, caught {proc_out}, trying new port..."
            )
            proc, port, proc_out = start_server()
    print(proc_out)

    server_pid = proc.pid
    PROCS.append(server_pid)
    logging.info(
        f"running new instance on {port=} with {server_pid=}:"
        + " ".join(launch_server + [str(port)])
    )

    ws_url = get_socket_url(port)
    return ServerInstanceInfo(port, ws_url, server_pid)


async def run_servers(num_inst: int) -> list[ServerInstanceInfo]:
    servers_start_tasks = []

    async def run():
        server_info = await run_server_instance(should_start_server=False)
        servers_start_tasks.append(server_info)

    await asyncio.gather(*[run() for _ in range(num_inst)])

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
def server_manager(server_queue: Queue[ServerInstanceInfo]):
    global PROCS

    servers_info = asyncio.run(run_servers(GeneralConfig.SERVER_COUNT))

    for server_info in servers_info:
        server_queue.put(server_info)
    try:
        yield
    finally:
        for proc in list(PROCS):
            kill_process(proc)
        PROCS = []


def main():
    global SERVER_INSTANCES, PROCS, RESULTS

    # Queue[ServerInstanceInfo]
    SERVER_INSTANCES = Queue()
    PROCS = []
    RESULTS = []

    with server_manager(SERVER_INSTANCES):
        app = web.Application()
        app.add_routes(routes)
        web.run_app(app, port=BrokerConfig.BROKER_PORT)


if __name__ == "__main__":
    main()
