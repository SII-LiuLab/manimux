"""Install the dashboard's page-local keyboard bindings on each live Viser client."""

import json
from pathlib import Path

import viser
from viser import _messages


def install_shortcuts(server: viser.ViserServer, buttons: dict[str, viser.GuiButtonHandle]) -> None:
    button_ids = {}
    for key, button in buttons.items():
        button.label = f"{button.label} ({key})"
        button_ids[key.lower()] = button._impl.uuid

    script = Path(__file__).with_name("shortcuts.js").read_text(encoding="utf-8")
    message = _messages.RunJavascriptMessage(
        source=script + f"\ninstallShortcuts({json.dumps(button_ids)});"
    )

    @server.on_client_connect
    def on_connect(client: viser.ClientHandle) -> None:
        client._websock_connection.queue_message(message)
