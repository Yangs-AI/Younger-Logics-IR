import os
import glob
import json
import argparse


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch as th
import onnxruntime as ort
import gymnasium as gym
from huggingface_hub import hf_hub_download

# Import all supported algorithms from stable_baselines3 and its contrib
from stable_baselines3 import PPO, SAC, DQN, A2C, TD3, DDPG


#Core Engine: PureMathWrapper
#Bypasses the strict probability distribution check in PyTorch 2.0+ Dynamo 
#by manually mapping the pure mathematical tensor flow.
class PureMathWrapper(th.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.algo = model.__class__.__name__
        self.action_space = model.action_space
        
        # Security interceptor: Block RNNs and Mask mechanisms to prevent ONNX graph explosion
        if "recurrent" in self.algo.lower() or "maskable" in self.algo.lower():
            raise NotImplementedError(f"Security block: Extraction for {self.algo} (RNN/Mask) is not supported.")
        
        # Precisely extract the core neural organ based on the algorithm family
        if self.algo in ["SAC", "TD3", "DDPG", "TQC"]:
            self.net = model.policy.actor
        elif self.algo in ["DQN", "QRDQN"]:
            self.net = model.policy.q_net
        else: 
            # Covers PPO, A2C, TRPO, etc.
            self.net = model.policy
            
        # Detect multi-modal input (Dict observation space)
        self.is_dict = isinstance(model.observation_space, gym.spaces.Dict)
        if self.is_dict:
            self.keys = list(model.observation_space.spaces.keys())

    def forward(self, *args):
        # Reconstruct the dictionary if multi-modal input is detected
        if self.is_dict:
            obs = {k: v for k, v in zip(self.keys, args)}
        else:
            obs = args[0]
            
        
        
        # Q-Network family: Pure mathematical mapping, pass through directly
        if self.algo in ["DQN", "QRDQN", "TD3", "DDPG"]:
            return self.net(obs)
            
        #Continuous control with distributions-SAC/TQC: 
        #Manually extract features and map to mean actions to avoid Normal dist instantiation.
        elif self.algo in ["SAC", "TQC"]:
            features = self.net.extract_features(obs, self.net.features_extractor)
            latent_pi = self.net.latent_pi(features)
            # Apply tanh to bound the actions within [-1, 1]
            return th.tanh(self.net.mu(latent_pi))
            
        #Mixed action space family (PPO, A2C): 
        #Manually extract features to avoid Categorical/Normal dist instantiation.
        else:
            features = self.net.extract_features(obs)
            if self.net.share_features_extractor:
                latent_pi, _ = self.net.mlp_extractor(features)
            else:
                latent_pi = self.net.mlp_extractor.forward_actor(features[0])
                
            mean_actions = self.net.action_net(latent_pi)
            
            # Discrete space requires argmax for the highest score; Continuous space outputs directly
            if isinstance(self.action_space, gym.spaces.Discrete):
                return th.argmax(mean_actions, dim=1)
            else:
                return mean_actions

def dynamic_load_model(repo_id, zip_path):
    """
    Attempts to load the model by guessing its algorithm based on the repository name,
    falling back to trial-and-error if the name is ambiguous.
    """
    ALGO_MAP = {
        "ppo": PPO, "a2c": A2C, "dqn": DQN, 
        "sac": SAC, "td3": TD3, "ddpg": DDPG
    }
    
    # Custom objects to bypass memory/timeout compatibility errors in older SB3 models
    custom_objects = {"optimize_memory_usage": False, "handle_timeout_termination": False}
    repo_lower = repo_id.lower()
    
    #Match by repo name
    for algo_name, algo_class in ALGO_MAP.items():
        if algo_name in repo_lower:
            return algo_class.load(zip_path, device="cpu", custom_objects=custom_objects)
            
    #Brute-force trial-and-error
    for algo_class in ALGO_MAP.values():
        try:
            return algo_class.load(zip_path, device="cpu", custom_objects=custom_objects)
        except Exception:
            continue
            
    raise ValueError(f"Failed to identify the algorithm type for {repo_id}.")


def process_and_export(repo_id, zip_filename, output_dir, save_name, current_idx, total_count):
    print(f"\n" + "-"*60)
    print(f" [{current_idx}/{total_count}] Processing model: {repo_id}")
    
    # Construct the final absolute export path
    export_path = os.path.join(output_dir, f"{save_name}.onnx")
    
    #Skip if the DAG already exists
    if os.path.exists(export_path):
        print(f"already exists, skip: {export_path}")
        return

    try:
        #Download from hf
        zip_path = hf_hub_download(repo_id=repo_id, filename=zip_filename)
        
        #Load and identify algorithm
        model = dynamic_load_model(repo_id, zip_path)
        algo_name = model.__class__.__name__
        print(f"      [+] Algorithm identified: {algo_name}")
        
        #Wrap and secure the model
        wrapped_model = PureMathWrapper(model)
        wrapped_model.eval() 
        
        #Construct dummy input tensors dynamically
        obs_space = model.observation_space
        if wrapped_model.is_dict:
            dummy_input = tuple(th.randn(1, *obs_space.spaces[k].shape) for k in wrapped_model.keys)
            input_names = [f"input_{k}" for k in wrapped_model.keys]
        else:
            dummy_input = (th.randn(1, *obs_space.shape), )
            input_names = ["input_obs"]

        #Export to ONNX DAG
        th.onnx.export(
            wrapped_model, dummy_input, export_path,
            opset_version=14, input_names=input_names, output_names=["output_action"]
        )
        print(f"DAG exported: {export_path}")
        
    except Exception as e:
        print(f"Failed to process model, auto-skipping. Caused by: {e}")


def scan_and_process_all(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    print(f"Discovered {len(json_files)} JSON files in directory: {input_folder}")
    
    sb3_models = []
    seen_ids = set() 
    
    # Aggregate and filter all SB3 models from sharded JSON files
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for model in data:
                    model_id = model.get("id")
                    tags = model.get("tags", [])
                    
                    if "stable-baselines3" in tags and model_id not in seen_ids:
                        sb3_models.append(model)
                        seen_ids.add(model_id)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            
    total_models = len(sb3_models)
    print(f"\nAggregation complete. Found {total_models} SB3 models. Target output dir: {output_folder}")
    
    if total_models == 0:
        print("No SB3 models found. Terminating pipeline.")
        return
        
    #Execute the processing pipeline
    for i, model_info in enumerate(sb3_models):
        repo_id = model_info["id"]
        
        #Locate the exact .zip file from the siblings list
        zip_filename = None
        for sibling in model_info.get("siblings", []):
            if sibling["rfilename"].endswith(".zip"):
                zip_filename = sibling["rfilename"]
                break
                
        if not zip_filename:
            print(f"\n[-] No .zip file found in repo {repo_id}, skipping.")
            continue
            
        # Sanitize filename by replacing '/' with '_'
        save_name = f"model_{repo_id.replace('/', '_')}"
        process_and_export(repo_id, zip_filename, output_folder, save_name, current_idx=i+1, total_count=total_models)


if __name__ == "__main__":
    #Standard CLI for engineering operations
    parser = argparse.ArgumentParser(description="Batch convert Hugging Face SB3 models to pure ONNX DAGs")
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="./Younger_Data", 
        help="Directory containing sharded module_info.json files (Default: ./Younger_Data)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./onnx_outputs", 
        help="Target directory to save extracted ONNX graphs (Default: ./onnx_outputs)"
    )
    
    args = parser.parse_args()
    
    #Initialize pipeline
    scan_and_process_all(args.input_dir, args.output_dir)