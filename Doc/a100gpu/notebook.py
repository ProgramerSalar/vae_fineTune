
import matplotlib.pyplot as plt
import re

def parse_and_plot(log_file_path):
    # Data storage
    data = {
        'step': [],
        'rec_loss': [],
        'perception_loss': [],
        'kl_loss': [],
        'vae_loss': [],
        'disc_loss': [],
        'd_weight': [],
        'grad_norm': []
    }

    # Read the file
    with open(log_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if 'Epoch: [' not in line: continue

        # 1. Extract Global Step
        # Format: Epoch: [E] [ S/T]
        epoch_match = re.search(r'Epoch: \[(\d+)\]', line)
        step_match = re.search(r'\[\s*(\d+)/(\d+)\]', line)
        
        if epoch_match and step_match:
            epoch = int(epoch_match.group(1))
            step = int(step_match.group(1))
            total_steps = int(step_match.group(2))
            global_step = epoch * total_steps + step
            data['step'].append(global_step)

            # 2. Extract Metrics (Helper function)
            def get_val(key):
                # Pattern searches for "key: value (" 
                match = re.search(key + r':\s*([-\d\.]+)\s*\(', line)
                return float(match.group(1)) if match else None

            data['rec_loss'].append(get_val('rec_loss'))
            data['perception_loss'].append(get_val('perception_loss'))
            data['kl_loss'].append(get_val('kl_loss'))
            data['vae_loss'].append(get_val('vae_loss'))
            data['disc_loss'].append(get_val('disc_loss'))
            data['d_weight'].append(get_val('d_weight'))
            data['grad_norm'].append(get_val('grad_norm'))

    # 3. Plotting
    metrics = ['rec_loss', 'perception_loss', 'd_weight', 'grad_norm']
    plt.figure(figsize=(15, 10))

    for i, metric in enumerate(metrics):
        # Filter None values
        valid_points = [(s, v) for s, v in zip(data['step'], data[metric]) if v is not None]
        if valid_points:
            steps, values = zip(*valid_points)
            plt.subplot(2, 2, i+1)
            plt.plot(steps, values)
            plt.title(metric.upper())
            plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('loss_graphs.png')
    print("Graph saved as loss_graphs.png")

if __name__ == "__main__":

# 
    txt_path = "/home/manish/Desktop/projects/vae_fineTune/Doc/a100gpu/batch_size=1/test.txt"
    parse_and_plot(txt_path)