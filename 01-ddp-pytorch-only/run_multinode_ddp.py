#!/usr/bin/env python3
"""
Multi-Node DDP Launcher

Uses Ray ONLY to orchestrate launching torchrun on multiple nodes.
The actual training uses vanilla PyTorch DDP - NOT Ray Train!

This demonstrates:
1. Multi-node DDP requires orchestration infrastructure
2. You need to coordinate MASTER_ADDR across nodes
3. You need to launch torchrun simultaneously on all nodes
"""

import argparse
import os
import subprocess
import ray


def get_node_ip():
    import socket
    return socket.gethostbyname(socket.gethostname())


@ray.remote(num_gpus=4)
class DDPNode:
    """Runs torchrun on a GPU node."""

    def __init__(self, node_rank, num_nodes, master_addr, master_port):
        self.node_rank = node_rank
        self.num_nodes = num_nodes
        self.master_addr = master_addr
        self.master_port = master_port
        self.node_ip = get_node_ip()

    def get_info(self):
        return {"rank": self.node_rank, "ip": self.node_ip}

    def run_torchrun(self, script_path, epochs, batch_size, lr):
        """Launch torchrun - this runs vanilla PyTorch DDP, NOT Ray Train!"""
        cmd = [
            "torchrun",
            f"--nnodes={self.num_nodes}",
            "--nproc_per_node=4",
            f"--node_rank={self.node_rank}",
            f"--master_addr={self.master_addr}",
            f"--master_port={self.master_port}",
            script_path,
            f"--epochs={epochs}",
            f"--batch-size={batch_size}",
            f"--lr={lr}",
        ]
        print(f"[Node {self.node_rank}] {self.node_ip}: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=False, cwd=os.path.dirname(script_path))
        return {"rank": self.node_rank, "success": result.returncode == 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-nodes", type=int, default=2)
    args = parser.parse_args()

    ray.init()

    # Get GPU worker nodes dynamically
    nodes = ray.nodes()
    gpu_nodes = [n["NodeManagerAddress"] for n in nodes
                 if n["Alive"] and n.get("Resources", {}).get("GPU", 0) >= 4]

    num_nodes = min(args.num_nodes, len(gpu_nodes))
    print(f"Found {len(gpu_nodes)} GPU nodes, using {num_nodes}")

    if num_nodes < 1:
        print("ERROR: No GPU nodes available!")
        return

    # First node is master
    master_addr = gpu_nodes[0]
    master_port = 29500
    script_path = "/mnt/cluster_storage/vhol-ddp/train_ddp.py"

    print(f"\nMaster: {master_addr}:{master_port}")
    print(f"Nodes: {gpu_nodes[:num_nodes]}")
    print(f"Script: {script_path}\n")

    # Create actors on each node
    actors = []
    for i in range(num_nodes):
        actor = DDPNode.options(
            resources={f"node:{gpu_nodes[i]}": 0.001}
        ).remote(i, num_nodes, master_addr, master_port)
        actors.append(actor)

    # Verify placement
    infos = ray.get([a.get_info.remote() for a in actors])
    print("Node placement:")
    for info in infos:
        print(f"  Rank {info['rank']}: {info['ip']}")

    # Launch training on all nodes simultaneously
    print("\nLaunching torchrun on all nodes...\n")
    futures = [a.run_torchrun.remote(script_path, args.epochs, args.batch_size, args.lr)
               for a in actors]

    results = ray.get(futures)

    print("\n" + "=" * 60)
    success = all(r["success"] for r in results)
    print(f"Training {'COMPLETED' if success else 'FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
