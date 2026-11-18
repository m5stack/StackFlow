#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 M5Stack Technology CO LTD
#
# SPDX-License-Identifier: MIT
import argparse
import json
import socket
import uuid
from typing import Any, Dict, Optional


class TcpJsonClient:
    def __init__(self, host="127.0.0.1", port=10001, timeout=60.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self._rx_buf = b""

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(None)

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def send_json(self, payload: Dict[str, Any]):
        assert self.sock is not None
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self.sock.sendall(data)

    def recv_json(self) -> Dict[str, Any]:
        assert self.sock is not None
        while True:
            if b"\n" in self._rx_buf:
                line, self._rx_buf = self._rx_buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                return json.loads(line.decode("utf-8", errors="replace"))
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Server closed connection")
            self._rx_buf += chunk

    def recv_json_stream(self):
        while True:
            yield self.recv_json()


def new_request_id() -> str:
    return str(uuid.uuid4())


def assert_ok(resp: Dict[str, Any]):
    err = resp.get("error") or {}
    code = err.get("code", 0)
    msg = err.get("message", "")
    if code != 0:
        raise RuntimeError(f"Server error code={code}, message={msg}")


def make_setup_kws(kws_text="HI PYRAMID") -> Dict[str, Any]:
    return {
        "request_id": new_request_id(),
        "work_id": "kws",
        "action": "setup",
        "object": "kws.setup",
        "data": {
            "model": "kws-ax650",
            "response_format": "kws.bool",
            "input": ["buttons_thread", "sys.pcm"],
            "enoutput": True,
            "kws": kws_text,
            "enwake_audio": True,
        },
    }


def make_setup_asr(kws_work_id: str) -> Dict[str, Any]:
    return {
        "request_id": new_request_id(),
        "work_id": "asr",
        "action": "setup",
        "object": "asr.setup",
        "data": {
            "model": "sense-voice-small-10s-ax650",
            "response_format": "asr.utf-8.stream",
            "input": [kws_work_id, "sys.pcm"],
            "enoutput": True,
        },
    }


def make_setup_llm(kws_work_id: str, asr_work_id: str) -> Dict[str, Any]:
    return {
        "request_id": new_request_id(),
        "work_id": "llm",
        "action": "setup",
        "object": "llm.setup",
        "data": {
            "model": "qwen2.5-0.5B-Int4-ax650",
            "response_format": "llm.utf-8.stream",
            "input": [kws_work_id, asr_work_id],
            "enoutput": True,
            "max_token_len": 256,
            "prompt": "You are a knowledgeable assistant capable of answering various questions and providing information. Respond exclusively in English and do not use any other language.",
        },
    }


def make_setup_melotts(kws_work_id: str, llm_work_id: str) -> Dict[str, Any]:
    return {
        "request_id": new_request_id(),
        "work_id": "melotts",
        "action": "setup",
        "object": "melotts.setup",
        "data": {
            "model": "melotts-en-default-ax650",
            "response_format": "sys.pcm",
            "input": [kws_work_id, llm_work_id],
            "enoutput": False,
        },
    }


def make_exit(work_id: str) -> Dict[str, Any]:
    return {"request_id": new_request_id(), "work_id": work_id, "action": "exit"}


def make_trigger(work_id: str) -> Dict[str, Any]:
    return {"request_id": new_request_id(), "work_id": work_id, "action": "trigger"}


class App:
    def __init__(self, host: str, port: int, kws_text: str):
        self.client = TcpJsonClient(host, port)
        self.kws_text = kws_text
        self.workids: Dict[str, Optional[str]] = {"kws": None, "asr": None, "llm": None, "melotts": None}
        self.created = []
        self._asr_text = ""
        self._llm_text = ""

    def _add_created(self, wid: Optional[str]):
        if wid and wid not in self.created:
            self.created.append(wid)

    def setup_pipeline(self):
        self.client.send_json(make_setup_kws(self.kws_text))
        resp = self.client.recv_json()
        assert_ok(resp)
        self.workids["kws"] = resp.get("work_id")
        self._add_created(self.workids["kws"])

        self.client.send_json(make_setup_asr(self.workids["kws"]))
        resp = self.client.recv_json()
        assert_ok(resp)
        self.workids["asr"] = resp.get("work_id")
        self._add_created(self.workids["asr"])

        self.client.send_json(make_setup_llm(self.workids["kws"], self.workids["asr"]))
        resp = self.client.recv_json()
        assert_ok(resp)
        self.workids["llm"] = resp.get("work_id")
        self._add_created(self.workids["llm"])

        self.client.send_json(make_setup_melotts(self.workids["kws"], self.workids["llm"]))
        resp = self.client.recv_json()
        assert_ok(resp)
        self.workids["melotts"] = resp.get("work_id")
        self._add_created(self.workids["melotts"])

    def handle_message(self, msg: Dict[str, Any]):
        assert_ok(msg)
        obj = msg.get("object")
        data = msg.get("data")

        if obj == "kws.bool" and data is True:
            self._asr_text = ""
            self._llm_text = ""
            print("\n[kws.trigger]")
            return

        if obj == "asr.utf-8.stream" and isinstance(data, dict):
            delta = data.get("delta", "")
            if delta:
                self._asr_text += delta
                print(delta, end="", flush=True)
            if data.get("finish") is True:
                print()
            return

        if obj == "llm.utf-8.stream" and isinstance(data, dict):
            delta = data.get("delta", "")
            if delta:
                self._llm_text += delta
                print(delta, end="", flush=True)
            if data.get("finish") is True:
                print()
            return

    def run(self):
        self.client.connect()
        try:
            self.setup_pipeline()
            for msg in self.client.recv_json_stream():
                self.handle_message(msg)
        finally:
            for wid in reversed(self.created):
                try:
                    self.client.send_json(make_exit(wid))
                except Exception:
                    pass
            self.client.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10001)
    ap.add_argument("--kws", default="HI PYRAMID")
    args = ap.parse_args()
    App(args.host, args.port, args.kws).run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass