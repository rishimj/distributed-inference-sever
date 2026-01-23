#!/usr/bin/env python3
"""
PACE ICE Cluster Deployment Script

Deploys disaggregated inference system on Georgia Tech PACE ICE cluster
with optimal resource allocation and SLURM job management.
"""

import asyncio
import argparse
import logging
import time
import subprocess
import yaml
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
import os


class SLURMJobManager:
    """
    Manages SLURM job submission and monitoring for PACE ICE deployment.
    """
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.active_jobs: Dict[str, str] = {}  # job_name -> job_id
        
    def create_prefill_job_script(self, 
                                 nodes: int,
                                 model_name: str,
                                 partition: str = "gpu-a100") -> str:
        """
        Create SLURM job script for prefill workers.
        Optimized for high-compute A100 nodes.
        """
        return f"""#!/bin/bash
#SBATCH --job-name=prefill-workers
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/prefill-%j.out
#SBATCH --error=logs/prefill-%j.err
#SBATCH --export=ALL

# Load modules
module load cuda/11.8
module load python/3.10

# Activate virtual environment
source venv/bin/activate

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export NCCL_DEBUG=INFO
export PYTHONPATH={self.project_path}:$PYTHONPATH

# Configure UCX for InfiniBand optimization on PACE ICE
export UCX_NET_DEVICES=mlx5_0:1
export UCX_TLS=rc_mlx5,ud_mlx5,mm,shm
export UCX_RNDV_SCHEME=put_zcopy
export UCX_RNDV_THRESH=8192
export UCX_MAX_RNDV_RAILS=1
export UCX_MM_TLS=posix

# InfiniBand performance tuning
export IBV_FORK_SAFE=1
export RDMA_CORE_ROOT=/usr

# Get node information
NODE_NAME=$(hostname)
NODE_ID="prefill-${{SLURM_PROCID}}-${{NODE_NAME}}"
PORT=$((8000 + $SLURM_PROCID))

echo "Starting prefill worker: $NODE_ID on port $PORT"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Node: $NODE_NAME"

# Start prefill worker
cd {self.project_path}
python -m src.workers.prefill_worker \\
    --node-id "$NODE_ID" \\
    --host 0.0.0.0 \\
    --port $PORT \\
    --model "{model_name}" \\
    --max-seqs 16 \\
    --max-len 4096

echo "Prefill worker $NODE_ID shutting down"
"""
    
    def create_decode_job_script(self, 
                                nodes: int,
                                model_name: str,
                                partition: str = "gpu-rtx") -> str:
        """
        Create SLURM job script for decode workers.
        Optimized for high-memory RTX nodes.
        """
        return f"""#!/bin/bash
#SBATCH --job-name=decode-workers
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/decode-%j.out
#SBATCH --error=logs/decode-%j.err
#SBATCH --export=ALL

# Load modules
module load cuda/11.8
module load python/3.10

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export NCCL_DEBUG=INFO
export PYTHONPATH={self.project_path}:$PYTHONPATH

# Configure UCX for InfiniBand
export UCX_NET_DEVICES=mlx5_0:1
export UCX_TLS=rc_mlx5,ud_mlx5,mm,shm
export UCX_RNDV_SCHEME=put_zcopy

# Get node information
NODE_NAME=$(hostname)
NODE_ID="decode-${{SLURM_PROCID}}-${{NODE_NAME}}"
PORT=$((8100 + $SLURM_PROCID))

echo "Starting decode worker: $NODE_ID on port $PORT"
echo "GPU: $CUDA_VISIBLE_DEVICES" 
echo "Node: $NODE_NAME"

# Start decode worker
cd {self.project_path}
python -m src.workers.decode_worker \\
    --node-id "$NODE_ID" \\
    --host 0.0.0.0 \\
    --port $PORT \\
    --model "{model_name}" \\
    --max-seqs 64 \\
    --max-len 8192

echo "Decode worker $NODE_ID shutting down"
"""
    
    def create_gateway_job_script(self, model_name: str) -> str:
        """
        Create SLURM job script for gateway coordinator.
        Runs on CPU node for orchestration.
        """
        return f"""#!/bin/bash
#SBATCH --job-name=inference-gateway
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/gateway-%j.out
#SBATCH --error=logs/gateway-%j.err
#SBATCH --export=ALL

# Load modules
module load python/3.10

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH={self.project_path}:$PYTHONPATH

echo "Starting inference gateway"
echo "Node: $(hostname)"

# Start gateway server
cd {self.project_path}
python scripts/start_gateway.py \\
    --model "{model_name}" \\
    --host 0.0.0.0 \\
    --port 8080 \\
    --config-file cluster_config.yaml

echo "Gateway shutting down"
"""
    
    async def submit_job(self, job_script: str, job_name: str) -> str:
        """Submit job script to SLURM and return job ID."""
        # Write job script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.slurm', delete=False) as f:
            f.write(job_script)
            script_path = f.name
        
        try:
            # Submit job
            result = subprocess.run(
                ['sbatch', script_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse job ID from output: "Submitted batch job 123456"
            job_id = result.stdout.strip().split()[-1]
            self.active_jobs[job_name] = job_id
            
            logging.info(f"Submitted {job_name} with job ID {job_id}")
            return job_id
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to submit job {job_name}: {e}")
            raise
        finally:
            # Clean up temporary file
            os.unlink(script_path)
    
    async def wait_for_job_start(self, job_id: str, timeout: int = 300) -> bool:
        """Wait for job to start running."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = subprocess.run(
                    ['squeue', '-j', job_id, '-h', '-o', '%T'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                status = result.stdout.strip()
                
                if status == 'RUNNING':
                    logging.info(f"Job {job_id} is now running")
                    return True
                elif status in ['FAILED', 'CANCELLED', 'TIMEOUT']:
                    logging.error(f"Job {job_id} failed with status: {status}")
                    return False
                
                # Wait and check again
                await asyncio.sleep(5)
                
            except subprocess.CalledProcessError:
                # Job might not exist anymore (finished quickly)
                break
        
        logging.warning(f"Job {job_id} did not start within {timeout} seconds")
        return False
    
    async def get_job_nodes(self, job_id: str) -> List[str]:
        """Get list of nodes allocated to a job."""
        try:
            result = subprocess.run(
                ['squeue', '-j', job_id, '-h', '-o', '%N'],
                capture_output=True,
                text=True,
                check=True
            )
            
            nodelist = result.stdout.strip()
            if not nodelist:
                return []
            
            # Parse SLURM nodelist format (e.g., "node[001-004,010]")
            # This is a simplified parser - real implementation would be more robust
            nodes = []
            if '[' in nodelist:
                # Handle range format
                base = nodelist.split('[')[0]
                ranges = nodelist.split('[')[1].split(']')[0]
                
                for range_part in ranges.split(','):
                    if '-' in range_part:
                        start, end = range_part.split('-')
                        for i in range(int(start), int(end) + 1):
                            nodes.append(f"{base}{i:03d}")
                    else:
                        nodes.append(f"{base}{range_part}")
            else:
                nodes = [nodelist]
            
            return nodes
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to get nodes for job {job_id}: {e}")
            return []
    
    def cancel_all_jobs(self):
        """Cancel all active jobs."""
        for job_name, job_id in self.active_jobs.items():
            try:
                subprocess.run(['scancel', job_id], check=True)
                logging.info(f"Cancelled job {job_name} ({job_id})")
            except subprocess.CalledProcessError as e:
                logging.warning(f"Failed to cancel job {job_name}: {e}")
        
        self.active_jobs.clear()


class PACEICEDeployer:
    """
    Main deployment orchestrator for PACE ICE cluster.
    """
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.project_path = os.path.abspath('.')
        self.slurm_manager = SLURMJobManager(self.project_path)
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
    
    def load_config(self, config_path: str) -> Dict:
        """Load deployment configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def deploy_workers(self) -> Tuple[List[str], List[str]]:
        """
        Deploy prefill and decode workers across PACE ICE nodes.
        Returns (prefill_nodes, decode_nodes).
        """
        model_name = self.config.get('model_name', 'facebook/opt-125m')
        prefill_config = self.config.get('prefill', {})
        decode_config = self.config.get('decode', {})
        
        # Submit prefill workers
        prefill_nodes_count = prefill_config.get('nodes', 2)
        prefill_partition = prefill_config.get('partition', 'gpu-a100')
        
        logging.info(f"Deploying {prefill_nodes_count} prefill workers on {prefill_partition}")
        
        prefill_script = self.slurm_manager.create_prefill_job_script(
            prefill_nodes_count, model_name, prefill_partition
        )
        prefill_job_id = await self.slurm_manager.submit_job(prefill_script, 'prefill')
        
        # Submit decode workers
        decode_nodes_count = decode_config.get('nodes', 4)
        decode_partition = decode_config.get('partition', 'gpu-rtx')
        
        logging.info(f"Deploying {decode_nodes_count} decode workers on {decode_partition}")
        
        decode_script = self.slurm_manager.create_decode_job_script(
            decode_nodes_count, model_name, decode_partition
        )
        decode_job_id = await self.slurm_manager.submit_job(decode_script, 'decode')
        
        # Wait for jobs to start
        prefill_started = await self.slurm_manager.wait_for_job_start(prefill_job_id)
        decode_started = await self.slurm_manager.wait_for_job_start(decode_job_id)
        
        if not prefill_started or not decode_started:
            raise Exception("Failed to start worker jobs")
        
        # Get allocated nodes
        prefill_nodes = await self.slurm_manager.get_job_nodes(prefill_job_id)
        decode_nodes = await self.slurm_manager.get_job_nodes(decode_job_id)
        
        logging.info(f"Prefill nodes: {prefill_nodes}")
        logging.info(f"Decode nodes: {decode_nodes}")
        
        # Wait for workers to initialize
        await asyncio.sleep(30)
        
        return prefill_nodes, decode_nodes
    
    async def deploy_gateway(self, prefill_nodes: List[str], decode_nodes: List[str]) -> str:
        """
        Deploy gateway coordinator that orchestrates prefill/decode.
        """
        model_name = self.config.get('model_name', 'facebook/opt-125m')
        
        # Create cluster configuration file
        cluster_config = {
            'prefill_workers': [
                {'node_id': f'prefill-{i}-{node}', 'host': node, 'port': 8000 + i}
                for i, node in enumerate(prefill_nodes)
            ],
            'decode_workers': [
                {'node_id': f'decode-{i}-{node}', 'host': node, 'port': 8100 + i}
                for i, node in enumerate(decode_nodes)
            ],
            'model_name': model_name,
            'network_config': {
                'interconnect': 'infiniband',
                'bandwidth_gbps': 25
            }
        }
        
        with open('cluster_config.yaml', 'w') as f:
            yaml.dump(cluster_config, f)
        
        # Submit gateway job
        gateway_script = self.slurm_manager.create_gateway_job_script(model_name)
        gateway_job_id = await self.slurm_manager.submit_job(gateway_script, 'gateway')
        
        # Wait for gateway to start
        gateway_started = await self.slurm_manager.wait_for_job_start(gateway_job_id)
        
        if not gateway_started:
            raise Exception("Failed to start gateway job")
        
        gateway_nodes = await self.slurm_manager.get_job_nodes(gateway_job_id)
        gateway_node = gateway_nodes[0] if gateway_nodes else 'unknown'
        
        logging.info(f"Gateway deployed on node: {gateway_node}")
        return gateway_node
    
    async def run_deployment(self):
        """Run complete deployment process."""
        try:
            logging.info("Starting PACE ICE deployment...")
            
            # Deploy workers
            prefill_nodes, decode_nodes = await self.deploy_workers()
            
            # Deploy gateway
            gateway_node = await self.deploy_gateway(prefill_nodes, decode_nodes)
            
            logging.info("Deployment completed successfully!")
            logging.info(f"Gateway endpoint: http://{gateway_node}:8080")
            logging.info(f"Prefill workers: {len(prefill_nodes)} nodes")
            logging.info(f"Decode workers: {len(decode_nodes)} nodes")
            
            # Monitor deployment
            await self.monitor_deployment()
            
        except KeyboardInterrupt:
            logging.info("Deployment interrupted by user")
        except Exception as e:
            logging.error(f"Deployment failed: {e}")
            raise
        finally:
            logging.info("Cleaning up resources...")
            self.slurm_manager.cancel_all_jobs()
    
    async def monitor_deployment(self):
        """Monitor running deployment."""
        logging.info("Monitoring deployment (Ctrl+C to stop)")
        
        while True:
            try:
                # Check job status
                for job_name, job_id in self.slurm_manager.active_jobs.items():
                    result = subprocess.run(
                        ['squeue', '-j', job_id, '-h', '-o', '%T %R'],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        status_info = result.stdout.strip()
                        logging.info(f"Job {job_name} ({job_id}): {status_info}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except subprocess.CalledProcessError:
                # Job finished or failed
                continue


async def main():
    parser = argparse.ArgumentParser(description="Deploy disaggregated inference on PACE ICE")
    parser.add_argument('--config', required=True, help='Deployment configuration file')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('deployment.log'),
            logging.StreamHandler()
        ]
    )
    
    # Run deployment
    deployer = PACEICEDeployer(args.config)
    await deployer.run_deployment()


if __name__ == "__main__":
    asyncio.run(main())