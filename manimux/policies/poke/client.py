"""Stateless Poke completions; local reset fences requests without execution receipts."""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from manimux.policies.base import action_interval
from manimux.policies.capabilities import PolicyCapabilities, metadata_mismatches
from manimux.policies.diagnostics import save_response
from manimux.policies.poke.codec import PokeCodec


class PokePolicyModel:
    def __init__(self, config):
        self.config = config
        options = config["options"]
        allowed = {
            "model",
            "base_url",
            "api_key_env",
            "request_timeout_s",
            "image_max_size",
            "response_log_dir",
        }
        if set(options) - allowed:
            raise ValueError(f"Unsupported Poke options: {sorted(set(options) - allowed)}")
        self.model = options["model"]
        self.base_url = options["base_url"].rstrip("/")
        parsed = urllib.parse.urlsplit(self.base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Poke base_url must be an HTTPS URL without credentials/query")
        self.key_env = options.get("api_key_env", "PLATFORM_API_KEY")
        if not isinstance(self.key_env, str) or not self.key_env.isidentifier():
            raise ValueError("Poke api_key_env must name an environment variable")
        self.timeout = float(options.get("request_timeout_s", config["timeout_s"]))
        if not np.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Poke timeout must be finite and positive")
        if not np.isclose(action_interval(config), 1 / 30, rtol=0, atol=1e-9):
            raise ValueError("Poke absolute targets use the documented 30 Hz clock")
        self.response_log_dir = options.get("response_log_dir", "data/diagnostics/poke-api")
        self.codec = PokeCodec(config)
        self.session_id = None
        self._key = ""
        self._metadata = {}

        # Credentials must never follow a redirect to a different provider.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        self._http = urllib.request.build_opener(NoRedirect)

    def _request(self, path, payload=None):
        request = urllib.request.Request(
            self.base_url + path,
            data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )
        try:
            with self._http.open(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
                request_id = response.headers.get("X-Request-ID")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"HTTP {exc.code}: {body.replace(self._key, '[REDACTED]')}"
            ) from None
        return body, request_id

    def reset(self, session_id):
        self.close()
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Poke reset requires a nonempty session ID")
        self._key = os.environ.get(self.key_env, "")
        if not self._key:
            raise ValueError(f"Set {self.key_env} in the runtime environment")
        try:
            body, _ = self._request("/models")
            if self.model not in {item["id"] for item in json.loads(body)["data"]}:
                raise ValueError("Poke model is not available to this API account")
            body, _ = self._request(
                "/models/" + urllib.parse.quote(self.model, safe="") + "/schema"
            )
            schema = json.loads(body)
            self.codec.bind_schema(schema)
            self._metadata = {
                "server": "poke_api",
                "model": {
                    "checkpoint": self.model,
                    "schema_model": schema.get("model"),
                    "schema_revision": schema.get("revision"),
                    "action_semantics": "absolute_model_flange_xyz_xyzw",
                },
            }
            expected = self.config.get("expected_backend")
            if expected and (errors := metadata_mismatches(expected, self._metadata)):
                raise ValueError(f"Poke backend identity mismatch: {errors}")
            self.session_id = session_id
        except Exception as exc:
            message = str(exc).replace(self._key, "[REDACTED]")
            self.close()
            raise RuntimeError(f"Poke startup failed: {message}") from None

    def infer(self, request):
        if self.session_id is None or request.session_id != self.session_id:
            raise ValueError("Poke request does not belong to the active session")
        messages, observation = self.codec.encode(request)
        start = time.monotonic()
        raw, request_id, error = None, None, None
        try:
            raw, request_id = self._request(
                "/chat/completions",
                {
                    "model": self.model,
                    "messages": messages,
                },
            )
            raw = json.loads(raw)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}".replace(self._key, "[REDACTED]")
        timing = {
            "rtt_ms": (time.monotonic() - start) * 1000,
            "request_id": request_id,
            "mode": "stateless",
            "base_url": self.base_url,
        }
        path = save_response(
            self.response_log_dir,
            request,
            checkpoint=self.model,
            observation=observation,
            raw=raw,
            timing=timing,
            secret=self._key,
            error=error,
            provider="poke",
        )
        if error:
            raise RuntimeError(f"Poke inference failed: {error}; response_log={path}")
        try:
            result = self.codec.decode(raw)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"{exc}; response_log={path}") from None
        result["sampling_stats"]["poke_api"].update(timing, response_log=str(path))
        return result

    def capabilities(self):
        return PolicyCapabilities(
            sampling_modes=frozenset({"default"}),
            multi_inflight=False,
            backend_metadata=dict(self._metadata),
        )

    def close(self):
        self.session_id = None
        self._metadata = {}
        self._key = ""
        self.codec.schema = None


def build_model(config):
    return PokePolicyModel(config)
