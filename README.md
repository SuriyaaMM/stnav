# Stochastic Terrain Navigation 

- [Report](./reports/stnav_technical_report.pdf)
- [Presentation](./reports/stnav_presentation.pdf)

## Project Structure
```
stnav/
---- source/
-------- terrain.py # terrain generation & configurations
-------- network.py # actor-crictic network & ppo training loop
-------- main.py # main script (orchestrator)
-------- visualise.py # post train visualisation
-------- compute_profile.py # compute profile generation
-------- visualise_compute_profile.py # compute profile visualiation
```

## Installation Guide

### Pre-Requisites
- [uv](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1)

#### Clone the Repo
```bash
git clone https://github.com/SuriyaaMM/stnav
```
#### Synchronize UV
```bash
uv sync
```
#### Activate the Virtual Environment
```bash
source stnav/bin/activate
```

#### Run the training script
```bash
uv run source/main.py
```