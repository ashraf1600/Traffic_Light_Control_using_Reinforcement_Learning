# Traffic Light Control with Reinforcement Learning (DQ

This project implements a Traffic Light Control system using SUMO (Simulation of Urban MObility) and a Deep Q-Network (DQN) agent.

## Prerequisites

1.  **Python 3.x**
2.  **SUMO Traffic Simulator**: Download and install from [sumo.dlr.de](https://sumo.dlr.de/docs/Downloads.html).

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure SUMO**:
    The script attempts to automatically find your SUMO installation.
    
    *   **If you have installed SUMO**: The script should find it in standard locations (e.g., `C:\Program Files (x86)\Eclipse\Sumo`).
    *   **If it fails**: Open `sumo_traffic_rl.py` and set the `CUSTOM_SUMO_PATH` variable at the top:
        ```python
        CUSTOM_SUMO_PATH = r"C:\Path\To\Your\Sumo"
        ```

## Running the Project

Run the main script:

```bash
python sumo_traffic_rl.py
```

## Project Structure

*   `sumo_traffic_rl.py`: Main training script.
*   `cross3ltl.sumocfg`: SUMO configuration file.
*   `net.net.xml`: Network definition (intersections, lanes).
*   `input_routes.rou.xml`: Traffic flow definitions.

## Troubleshooting

*   **Unicode/Emoji Errors**: The script has been cleaned to work on standard Windows terminals.
*   **"SUMO_HOME not set"**: This means the script cannot find SUMO. Please double-check your installation path and update `CUSTOM_SUMO_PATH` if necessary.
