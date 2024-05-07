# server-less-fed-learn
B.Tech Project on the Federated Learning without the dedicated server.

## How to Run

- Generate dataset
    - Example for dataset generation
    ```bash
    # partition the CIFAR-10 according to Dir(0.1) for 100 clients
    python generate_data.py -d cifar10 -a 0.1 -cn 100
    ```

- run the main 
