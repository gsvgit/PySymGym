import argparse
import json
import logging
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from queue import Empty, Queue

import psutil
from aiohttp import web

from common.constants import SERVER_WORKING_DIR
from config import BrokerConfig, FeatureConfig, GeneralConfig, ServerConfig
from connection.broker_conn.classes import ServerInstanceInfo, Undefined, WSUrl

routes = web.RouteTableDef()
logging.basicConfig(
    level=GeneralConfig.LOGGER_LEVEL,
    filename="instance_manager.log",
    filemode="w",
    format="%(asctime)s - p%(process)d: %(name)s - [%(levelname)s]: %(message)s",
)


@routes.get("/get_ws")
async def dequeue_instance(request):
    try:
        server_info = SERVER_INSTANCES.get(block=False)
        assert server_info.pid is Undefined
        server_info = run_server_instance(
            port=server_info.port,
            start_server=FeatureConfig.ON_GAME_SERVER_RESTART.enabled,
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


def run_server_instance(port: int, start_server: bool) -> ServerInstanceInfo:
    launch_server = [
        "dotnet",
        "VSharp.ML.GameServer.Runner.dll",
        "--checkactualcoverage",
        "--port",
    ]
    ws_url = get_socket_url(port)
    if not start_server:
        return ServerInstanceInfo(port, ws_url, pid=Undefined)

    proc = subprocess.Popen(
        launch_server + [str(port)],
        stdout=subprocess.PIPE,
        start_new_session=True,
        cwd=SERVER_WORKING_DIR,
    )

    while True:
        out = proc.stdout.readline()
        if out and "Smooth!" in out.decode("utf-8"):
            print(out.decode("utf-8"), end="")
            break

    server_pid = proc.pid
    PROCS.append(server_pid)
    logging.info(
        f"running new instance on {port=} with {server_pid=}:"
        + f"{server_pid}: "
        + " ".join(launch_server + [str(port)])
    )

    ws_url = get_socket_url(port)
    return ServerInstanceInfo(port, ws_url, server_pid)


def run_servers(num_inst: int, start_port: int) -> list[ServerInstanceInfo]:
    servers_info = []
    for i in range(num_inst):
        server_info = run_server_instance(start_port + i, False)
        servers_info.append(server_info)

    return servers_info


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
    servers_info = run_servers(
        num_inst=GeneralConfig.SERVER_COUNT,
        start_port=ServerConfig.VSHARP_INSTANCES_START_PORT,
    )

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
    parser = argparse.ArgumentParser(description="V# instances launcher")
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        help="dont launch servers if set",
    )

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
