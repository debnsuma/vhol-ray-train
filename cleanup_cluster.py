"""
Ray Cluster Cleanup Script

This script clears GPU and CPU memory from all worker nodes in a Ray cluster.
Use this after training runs to ensure resources are freed.

Usage:
    python cleanup_cluster.py
"""

import ray
import gc
import torch


def cleanup_worker():
    """Clean up memory on a single worker node."""
    # Clear Python garbage
    gc.collect()

    # Clear CUDA memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return True


def cleanup_cluster():
    """Clear memory from all worker nodes in the Ray cluster."""
    # Initialize Ray if not already connected
    if not ray.is_initialized():
        ray.init()

    # Get cluster resources to determine number of nodes
    nodes = ray.nodes()
    num_nodes = len([n for n in nodes if n.get("Alive", False)])

    print(f"Cleaning up {num_nodes} nodes in the Ray cluster...")

    # Create a remote function to run cleanup on each node
    @ray.remote
    def remote_cleanup():
        return cleanup_worker()

    # Run cleanup on multiple workers to cover all nodes
    # Use num_cpus=0.01 to spread across nodes
    @ray.remote(num_cpus=0.01)
    def cleanup_on_node():
        cleanup_worker()
        return ray.get_runtime_context().get_node_id()

    # Launch cleanup tasks - more than nodes to ensure coverage
    num_tasks = max(num_nodes * 2, 4)
    futures = [cleanup_on_node.remote() for _ in range(num_tasks)]
    cleaned_nodes = set(ray.get(futures))

    print(f"Cleanup completed on {len(cleaned_nodes)} nodes")

    # Clear memory on the driver as well
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Cluster cleanup complete.")


if __name__ == "__main__":
    cleanup_cluster()
