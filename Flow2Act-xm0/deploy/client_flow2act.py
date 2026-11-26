# Copyright (C) 2026 Xiaomi Corporation.
import math
import os
import time
import pickle
import socket
import struct

import numpy as np
from PIL import Image



def resolve_policy_metadata_flow2act(metadata,*,replan_steps=None,memory_length=None):
    if metadata.get("response_type")!="policy_metadata" or metadata.get("history_protocol")!="real_decision_fifo_current_included":
        raise ValueError("Server did not provide compatible decision-policy metadata")
    if metadata.get("checkpoint_kind") not in {"trained_policy","mechanical_training_smoke"}:
        raise ValueError("Server policy purpose is unknown")
    n,m,length=(int(metadata[key]) for key in ("action_horizon","execute_horizon","memory_length"))
    if not 1<=m<=n<=50 or length < 1:
        raise ValueError("Server policy metadata contains an invalid N/M/L")
    if not isinstance(metadata.get("motion_3d_enabled"),bool):
        raise ValueError("Server motion_3d_enabled metadata must be boolean")
    if metadata.get("architecture_version") != "Flow2Act-XR0-checkpoint-v1":
        raise ValueError("Expected a Flow2Act policy; import external checkpoints before loading")
    if metadata.get("stage") not in ("baseline", "static", "motion2d", "motion3d"):
        raise ValueError("Unknown policy stage")
    future = metadata.get("future_horizon")
    if isinstance(future,bool) or not isinstance(future,int) or not 1<=future<=50:
        raise ValueError("Server future_horizon must be an integer in [1,50]")
    if metadata["motion_3d_enabled"] != (metadata["stage"] == "motion3d"):
        raise ValueError("Server stage and motion_3d_enabled metadata disagree")
    if metadata["stage"] in ("baseline", "static") and length != 1:
        raise ValueError("Baseline/static policies require memory_length=1")
    if replan_steps is not None and int(replan_steps)!=m:
        raise ValueError("Explicit replan_steps differs from the saved server execute_horizon")
    if memory_length is not None and int(memory_length)!=length:
        raise ValueError("Explicit memory_length differs from the saved server policy")
    return {**metadata,"action_horizon":n,"execute_horizon":m,"memory_length":length,"future_horizon":future}


class ClientFlow2Act:
    def __init__(self, host="localhost", port=10086):
        self.host = host
        self.port = port
        self._connect_with_retry(max_retries=10, retry_interval=1)
        print(f"Client connected to server at {self.host}:{self.port}.")

    def _connect_with_retry(self, max_retries=None, retry_interval=1):
        """Connect with retry logic. max_retries=None implies infinite."""
        retry_count = 0
        while True:
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((self.host, self.port))
                return
            except (ConnectionRefusedError, socket.error) as e:
                retry_count += 1
                time.sleep(retry_interval)
                if max_retries is not None and retry_count >= max_retries:
                    raise ConnectionError(f"Failed to connect to {self.host}:{self.port} after {retry_count} retries: {e}") from e

    def _send_with_length_prefix(self, data):
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        self.client_socket.sendall(struct.pack(">I", len(serialized)) + serialized)

    def _recv_with_length_prefix(self):
        len_data = b""
        while len(len_data) < 4:
            part = self.client_socket.recv(4-len(len_data))
            if not part:raise ConnectionError("Failed to receive response length prefix.")
            len_data += part
        data_len = struct.unpack(">I", len_data)[0]
        data = b""
        while len(data) < data_len:
            packet = self.client_socket.recv(data_len - len(data))
            if not packet:
                raise ConnectionError("Connection closed while receiving response.")
            data += packet
        return pickle.loads(data)

    def _request(self,data):
        for attempt in range(3):
            try:
                self._send_with_length_prefix(data)
                response = self._recv_with_length_prefix()
                return response
            except (OSError, ConnectionError):
                self.client_socket.close()
                if attempt == 2:raise
                self._connect_with_retry(max_retries=3,retry_interval=1)

    def policy_metadata(self,*,replan_steps=None,memory_length=None):
        response=self._request({"request_type":"metadata"})
        return resolve_policy_metadata_flow2act(response,replan_steps=replan_steps,memory_length=memory_length)

    def __call__(self, **data):
        response=self._request(data)
        if response["episode_id"] != data["episode_id"] or response["decision_id"] != data["decision_id"]:
            raise ValueError("RPC response does not match pending decision")
        return response["actions"]

    def close(self):
        self.client_socket.close()
        print("Client connection closed.")
