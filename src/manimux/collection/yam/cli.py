"""Start the YAM-specific GUI without connecting the robot on launch."""

from .backend import load_backend_config
from .config import build_station_config


def run_gui(args):
    import uvicorn

    from .gui.server import create_app

    config = build_station_config(args.station, args.cameras)
    load_backend_config(config, mock=args.mock or config.robot.type == "mock")
    app = create_app(
        config,
        mock=args.mock,
        station_path=args.station,
        cameras_path=args.cameras,
    )
    uvicorn.run(app, host=args.host, port=args.port)
