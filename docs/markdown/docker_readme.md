# ƒVDB Docker Help

Running a docker container is a great way to ensure that you have a consistent environment for building and running ƒVDB.

Our provided [`Dockerfile`](../../Dockerfile) constructs a Docker image which is ready to build ƒVDB.  The docker image is configured to install miniforge and the `fvdb` conda environment with all the dependencies needed to build and run ƒVDB.

## Setting up a Docker Container

Building and starting the docker image is done by running the following command from the fvdb directory:
```shell
docker compose run --rm fvdb-dev
```

When you are ready to build ƒVDB, run the following command within the docker container.  `TORCH_CUDA_ARCH_LIST` specifies which CUDA architectures to build for.
```shell
conda activate fvdb;
cd /workspace;
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX" \
./build.sh install verbose
```

If you've built an artifact inside the docker container using `./build.sh wheel`:
After you've built, if you want to extract the wheel artifact, from a separate terminal
(not running inside the container) you can use `docker cp` to extract the wheel. The wheel is
created with a name (in the container filesystem) like:
```
/workspace/dist/fvdb-0.2.1-cp312-cp312-linux_x86_64.whl
```
So we can use docker cp to extract it to a local directory. Use `docker ps -a` to determine what
the name of the running container is:
```
:~$ docker ps -a
CONTAINER ID   IMAGE           COMMAND       CREATED          STATUS          PORTS     NAMES
239232951e48   fvdb-fvdb-dev   "/bin/bash"   21 minutes ago   Up 21 minutes             fvdb-fvdb-dev-run-0615ba7e27fd
```

And then we can copy from that container to the local directory

```shell
docker cp fvdb-fvdb-dev-run-0615ba7e27fd:/workspace/dist/fvdb-0.2.1-cp312-cp312-linux_x86_64.whl .
```

### Workflow Examples

Here are the common cases.

### If you launched with `docker compose run --rm fvdb-dev`
- Exit the interactive shell (Ctrl-D or `exit`). The container is automatically removed.
- Rebuild image and relaunch:
```bash
cd ~/src/fvdb
docker compose build --no-cache fvdb-dev
docker compose run --rm fvdb-dev
```

### If you used `docker compose up -d fvdb-dev`
- Stop and remove the running container(s):
```bash
cd ~/src/fvdb
docker compose down
```
- Rebuild and start:
```bash
docker compose build --no-cache fvdb-dev
docker compose up -d fvdb-dev
```

### Optional deeper clean
- Also remove images and volumes:
```bash
docker compose down --rmi local --volumes
```

### Troubleshooting

* **docker daemon runtime/driver errors**
Errors like these:
```shell
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```
```shell
docker: Error response from daemon: Unknown runtime specified nvidia.
```
most likely indicate that the [`NVIDIA Container Toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) is not installed.
You can install it by following the [installation instructions here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

* **GPU access blocked by the operating system**
Errors like these:
```shell
Failed to initialize NVML: GPU access blocked by the operating system
Failed to properly shut down NVML: GPU access blocked by the operating system
```
may be solved by making a change to the file `/etc/nvidia-container-toolkit/config.toml`, setting
the property "no-cgroups" from "true" to "false":

```
# ...
[nvidia-container-cli]
# ...
# Below changed from previous no-cgroups = true
no-cgroups = false
# ...
```
