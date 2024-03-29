import atexit
import collections
import os
import random
import signal
import subprocess
import sys
import time
from typing import Any
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import carla
import numpy as np
import transforms3d.euler
from absl import logging

logging.set_verbosity(logging.DEBUG)


def setup(town: str, fps: int = 30, server_timestop: float = 30.0, client_timeout: float = 20.0, num_max_restarts: int = 10, playing=False):
    """Returns the `CARLA` `server`, `client` and `world`.

    Args:
        town: The `CARLA` town identifier.
        fps: The frequency (in Hz) of the simulation.
        server_timestop: The time interval between spawing the server
        and resuming program.
        client_timeout: The time interval before stopping
        the search for the carla server.
        num_max_restarts: Number of attempts to connect to the server.

    Returns:
        client: The `CARLA` client.
        world: The `CARLA` world.
        frame: The synchronous simulation time step ID.
        server: The `CARLA` server.
    """
    assert town in ("Town01", "Town02", "Town03", "Town04", "Town05")

    # The attempts counter.
    attempts = 0

    while attempts < num_max_restarts:
        print("*****attempts to connect to the server.******")
        logging.debug("{} out of {} attempts to setup the CARLA simulator".format(attempts + 1, num_max_restarts))

        # Random assignment of port.
        port = 2000

        # Start CARLA server.
        logging.debug("Inits a CARLA server at port={}".format(port))
        cmdline = (
            str(os.path.join("/home/cxw/Documents/CARLA_0.9.12", "CarlaUE4.sh"))
            + f" -carla-rpc-port={port}"
            + f" -prefernvidia -benchmark -fps 20 "
            + f" -quality-level=Low "
            + (f" -RenderOffScreen " if not playing else "")
        )
        print(cmdline)
        server = subprocess.Popen(
            cmdline,
            stdout=None,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            shell=True,
        )
        atexit.register(os.killpg, server.pid, signal.SIGKILL)
        time.sleep(4)

        # Connect client.
        logging.debug("Connects a CARLA client at port={}".format(port))
        try:
            client = carla.Client("localhost", port)  # pylint: disable=no-member
            client.set_timeout(client_timeout)
            client.load_world(map_name=town)
            world = client.get_world()
            # world.set_weather(carla.WeatherParameters.ClearNoon)  # pylint: disable=no-member
            frame = world.apply_settings(
                carla.WorldSettings(  # pylint: disable=no-member
                    synchronous_mode=True,
                    fixed_delta_seconds=0.05,
                    no_rendering_mode=True if not playing else False,
                )
            )
            logging.debug("Server version: {}".format(client.get_server_version()))
            logging.debug("Client version: {}".format(client.get_client_version()))
            return client, world, frame, server
        except RuntimeError as msg:
            logging.debug(msg)
            attempts += 1
            logging.debug("Stopping CARLA server at port={}".format(port))
            os.killpg(server.pid, signal.SIGKILL)
            atexit.unregister(lambda: os.killpg(server.pid, signal.SIGKILL))

    logging.debug("Failed to connect to CARLA after {} attempts".format(num_max_restarts))
    sys.exit()
