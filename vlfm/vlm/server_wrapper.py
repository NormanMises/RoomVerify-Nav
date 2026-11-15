# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import base64
import os
import random
import socket
import sys
import threading
import time
from typing import Any, Dict

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request


class VLMRequestError(RuntimeError):
    """Raised when a VLM HTTP request fails after retries.

    This replaces the previous behavior of calling exit(), so callers can catch
    the exception and shut down gracefully (e.g., VectorEnv.close()).
    """

    def __init__(self, message: str, *, url: str, attempts: int, last_exception: Exception) -> None:
        super().__init__(message)
        self.url = url
        self.attempts = attempts
        self.last_exception = last_exception


def _vlm_failure_should_pause() -> bool:
    """
    Controls what happens when a VLM HTTP request keeps failing.

    - If VLFM_VLM_FAILURE_MODE is set:
        - "wait"/"pause"/"debug": pause and wait for manual input, then retry
        - "raise"/"error": raise VLMRequestError immediately after retries
    - Otherwise: pause only when stdin is a TTY (interactive runs), and raise in
      non-interactive runs (e.g., batch jobs with closed stdin).
    """
    mode = os.environ.get("VLFM_VLM_FAILURE_MODE", "").strip().lower()
    if mode in ("wait", "pause", "debug", "interactive"):
        return True
    if mode in ("raise", "error", "abort", "fail"):
        return False
    return sys.stdin.isatty()


def _wait_for_manual_continue(err: VLMRequestError) -> bool:
    """
    Returns True to retry, False to raise.

    We only attempt to prompt if stdin is a TTY; otherwise we'd block forever or
    immediately hit EOF.
    """
    if not sys.stdin.isatty():
        return False

    print(
        "\n[VLFM] VLM request failed repeatedly; entering manual debug pause.\n"
        f"  url: {err.url}\n"
        f"  attempts: {err.attempts}\n"
        f"  last_error: {err.last_exception}\n"
        "Press Enter to retry, or type 'q' then Enter to abort.\n"
    )
    try:
        cmd = input("[VLFM] continue? [Enter=retry / q=abort] ").strip().lower()
    except EOFError:
        return False
    return cmd not in ("q", "quit", "exit", "abort", "raise")


class ServerMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def process_payload(self, payload: dict) -> dict:
        raise NotImplementedError


def host_model(model: Any, name: str, port: int = 5000) -> None:
    """
    Hosts a model as a REST API using Flask.
    """
    app = Flask(__name__)
    # 多个评测进程共用一个服务时串行推理，避免并发请求在同一个模型上互相抢占 GPU。
    inference_lock = threading.Lock()

    @app.route(f"/{name}", methods=["POST"])
    def process_request() -> Dict[str, Any]:
        payload = request.json
        with inference_lock:
            result = model.process_payload(payload)
        return jsonify(result)

    app.run(host="localhost", port=port)


def bool_arr_to_str(arr: np.ndarray) -> str:
    """Converts a boolean array to a string."""
    packed_str = base64.b64encode(arr.tobytes()).decode()
    return packed_str


def str_to_bool_arr(s: str, shape: tuple) -> np.ndarray:
    """Converts a string to a boolean array."""
    # Convert the string back into bytes using base64 decoding
    bytes_ = base64.b64decode(s)

    # Convert bytes to np.uint8 array
    bytes_array = np.frombuffer(bytes_, dtype=np.uint8)

    # Reshape the data back into a boolean array
    unpacked = bytes_array.reshape(shape)
    return unpacked


def image_to_str(img_np: np.ndarray, quality: float = 90.0) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    retval, buffer = cv2.imencode(".jpg", img_np, encode_param)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str


def str_to_image(img_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(img_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_np = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)
    return img_np


def send_request(url: str, **kwargs: Any) -> dict:
    max_attempts = max(1, int(os.environ.get("VLFM_VLM_MAX_RETRIES", "10")))
    while True:
        response: dict = {}
        total_start = time.time()
        for attempt in range(max_attempts):
            try:
                response = _send_request(url, **kwargs)
                return response
            except Exception as e:
                if attempt == max_attempts - 1:
                    elapsed = time.time() - total_start
                    msg = (
                        f"VLM request failed after {max_attempts} attempts "
                        f"({elapsed:.1f}s elapsed): {url}"
                    )
                    err = VLMRequestError(msg, url=url, attempts=max_attempts, last_exception=e)
                    if _vlm_failure_should_pause() and _wait_for_manual_continue(err):
                        # User requested to continue; start a new retry cycle.
                        break
                    raise err from e

                print(f"Error: {e}. Retrying in 20-30 seconds...")
                time.sleep(20 + random.random() * 10)


def _send_request(url: str, **kwargs: Any) -> dict:
    lockfiles_dir = "lockfiles"
    if not os.path.exists(lockfiles_dir):
        os.makedirs(lockfiles_dir)
    filename = url.replace("/", "_").replace(":", "_") + ".lock"
    filename = filename.replace("localhost", socket.gethostname())
    filename = os.path.join(lockfiles_dir, filename)
    try:
        while True:
            # Use a while loop to wait until this filename does not exist
            while os.path.exists(filename):
                # If the file exists, wait 50ms and try again
                time.sleep(0.05)

                try:
                    # If the file was last modified more than 120 seconds ago, delete it
                    if time.time() - os.path.getmtime(filename) > 120:
                        os.remove(filename)
                except FileNotFoundError:
                    pass

            rand_str = str(random.randint(0, 1000000))

            with open(filename, "w") as f:
                f.write(rand_str)
            time.sleep(0.05)
            try:
                with open(filename, "r") as f:
                    if f.read() == rand_str:
                        break
            except FileNotFoundError:
                pass

        # Create a payload dict which is a clone of kwargs but all np.array values are
        # converted to strings
        payload = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                payload[k] = image_to_str(v, quality=kwargs.get("quality", 90))
            else:
                payload[k] = v

        # Set the headers
        headers = {"Content-Type": "application/json"}

        start_time = time.time()
        while True:
            try:
                # 连接超时 1 秒；读超时 20 秒，等待排队中的推理完成，不在服务端忙时重复提交请求。
                resp = requests.post(url, headers=headers, json=payload, timeout=(1, 20))
                if resp.status_code == 200:
                    result = resp.json()
                    break
                else:
                    raise Exception("Request failed")
            except (
                requests.exceptions.Timeout,
                requests.exceptions.RequestException,
            ) as e:
                print(e)
                if time.time() - start_time > 20:
                    raise Exception("Request timed out after 20 seconds")

        try:
            # Delete the lock file
            os.remove(filename)
        except FileNotFoundError:
            pass

    except Exception as e:
        try:
            # Delete the lock file
            os.remove(filename)
        except FileNotFoundError:
            pass
        raise e

    return result
