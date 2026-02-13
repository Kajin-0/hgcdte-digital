~/fenicsx/run-dolfinx.sh

python3 your_simulation.py

mpirun -n 8 python3 your_simulation.py

docker run -it --rm \
  --user $(id -u):$(id -g) \
  -e HOME=/workspace \
  -e XDG_CACHE_HOME=/workspace/.cache \
  -e FFCX_CACHE_DIR=/workspace/.cache/ffcx \
  -e DIJITSO_CACHE_DIR=/workspace/.cache/dijitso \
  -v $HOME/fenicsx:/workspace \
  -w /workspace \
  dolfinx/dolfinx:stable
