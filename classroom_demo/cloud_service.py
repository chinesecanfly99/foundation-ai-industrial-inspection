"""Serve the shared Gradio workbench with separate storage for each browser."""
from dataclasses import dataclass, field
from http.cookies import CookieError, SimpleCookie
import json
import os
from pathlib import Path
import secrets
import shutil
from time import monotonic
from urllib.parse import parse_qs, urlsplit

from starlette.datastructures import Headers
from starlette.responses import JSONResponse

from app.app import create_app
from app.alt_ui import create_app as create_chinese
from app.server import create_server
from classroom_demo.cloud_models import use_small_cpu_backend
from industrial_inspection.workspace import active_results

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "gradio"
VISITORS = ROOT / ".cache" / "visitors"
COOKIE = "inspection_browser"
VISITOR_TTL = 7200
MAX_VISITORS = 128


def file_paths(value):
    if isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str):
            yield Path(path).resolve()
        for child in value.values():
            yield from file_paths(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from file_paths(child)


@dataclass
class Visitor:
    key: str
    last_seen: float = field(default_factory=monotonic)
    files: set[Path] = field(default_factory=set)
    sessions: set[str] = field(default_factory=set)

    @property
    def results(self):
        return VISITORS / self.key


class BrowserIsolation:
    def __init__(self, wrapped, public_files):
        self.wrapped = wrapped
        self.public_files = public_files
        self.visitors = {}
        self.sessions = {}
        self.last_sweep = 0.0
        self.secure = os.environ.get("INSPECTION_CLOUD_TEST") != "1"

    def get_visitor(self, headers):
        cookie = SimpleCookie()
        try:
            cookie.load(headers.get("cookie", ""))
        except CookieError:
            return None
        value = cookie.get(COOKIE)
        return self.visitors.get(value.value) if value else None

    def cleanup(self, now):
        if now - self.last_sweep < 300:
            return
        self.last_sweep = now
        expired = [key for key, visitor in self.visitors.items() if now - visitor.last_seen > VISITOR_TTL]
        stale_files = set()
        for key in expired:
            visitor = self.visitors.pop(key)
            stale_files.update(visitor.files)
            for session in visitor.sessions:
                self.sessions.pop(session, None)
            shutil.rmtree(visitor.results, ignore_errors=True)
        for visitor in self.visitors.values():
            stale_files.difference_update(visitor.files)
        for file in stale_files - self.public_files:
            if file.is_relative_to(CACHE):
                try:
                    file.unlink(missing_ok=True)
                except OSError:
                    continue

    def bind_session(self, visitor, session):
        if not session:
            return True
        if len(session) > 128:
            return False
        owner = self.sessions.get(session)
        if owner is not None:
            return owner == visitor.key
        if len(visitor.sessions) >= 32:
            return False
        self.sessions[session] = visitor.key
        visitor.sessions.add(session)
        return True

    def allows(self, visitor, path):
        return path in self.public_files or (visitor is not None and path in visitor.files)

    def scope_callbacks(self, blocks):
        original = blocks.process_api

        async def process_api(block_fn, inputs, **kwargs):
            request = kwargs.get("request")
            scope = request.request.scope if request and request.request else {}
            visitor = scope.get("state", {}).get("inspection_visitor")
            if visitor is None or not self.bind_session(visitor, kwargs.get("session_hash")):
                raise ValueError("The browser session expired. Reload the page.")
            if any(not self.allows(visitor, path) for path in file_paths(inputs)):
                raise ValueError("Upload or select the file in this browser first.")
            location = active_results.set(visitor.results)
            try:
                output = await original(block_fn, inputs, **kwargs)
                visitor.files.update(path for path in file_paths(output.get("data")) if path.is_relative_to(CACHE))
                return output
            finally:
                active_results.reset(location)

        blocks.process_api = process_api

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.wrapped(scope, receive, send)
            return
        path = scope["path"]
        headers = Headers(scope=scope)
        visitor = self.get_visitor(headers)
        new = False
        if scope["method"] == "GET" and path in {"/", "/zh/", "/config", "/zh/config"} and visitor is None:
            self.cleanup(monotonic())
            if len(self.visitors) >= MAX_VISITORS:
                await JSONResponse({"detail": "The classroom is full. Please try again later."}, 503)(scope, receive, send)
                return
            visitor = Visitor(secrets.token_hex(24))
            self.visitors[visitor.key] = visitor
            new = True
        if visitor is not None:
            visitor.last_seen = monotonic()
            scope.setdefault("state", {})["inspection_visitor"] = visitor

        async def deny(status=403):
            await JSONResponse({"detail": "This file or session is unavailable."}, status)(scope, receive, send)

        if "/file=" in path:
            asset = Path(path.split("/file=", 1)[1]).resolve()
            if not self.allows(visitor, asset):
                await deny()
                return
        if path.endswith("/queue/data"):
            session = parse_qs(scope.get("query_string", b"").decode()).get("session_hash", [""])[0]
            if visitor is None or not session or not self.bind_session(visitor, session):
                await deny()
                return

        uploading = path.endswith("/gradio_api/upload") and scope["method"] == "POST"
        if "/gradio_api/" in path and scope["method"] == "POST":
            origin = headers.get("origin")
            if visitor is None or (origin and urlsplit(origin).netloc != headers.get("host")):
                await deny()
                return
        if uploading and int(headers.get("content-length", "0")) > 105_000_000:
            await deny(413)
            return
        status = 0

        async def guarded_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                if new:
                    value = f"{COOKIE}={visitor.key}; Path=/; HttpOnly; SameSite=Lax; Max-Age={VISITOR_TTL}"
                    if self.secure:
                        value += "; Secure"
                    message.setdefault("headers", []).append((b"set-cookie", value.encode()))
                if "/file=" in path:
                    message.setdefault("headers", []).append((b"cache-control", b"private, no-store"))
            elif uploading and status == 200 and message["type"] == "http.response.body":
                uploads = json.loads(message.get("body", b"[]"))
                for name in uploads:
                    file = Path(name).resolve()
                    if file.is_relative_to(CACHE):
                        visitor.files.add(file)
            await send(message)

        await self.wrapped(scope, receive, guarded_send)


use_small_cpu_backend()
english, chinese = create_app(), create_chinese()
app = BrowserIsolation(create_server(english, chinese, default_language="en"),
                       set(file_paths(english.config)) | set(file_paths(chinese.config)))
app.scope_callbacks(english)
app.scope_callbacks(chinese)
