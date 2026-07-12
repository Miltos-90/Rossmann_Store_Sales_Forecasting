#!/bin/bash
# 1. Install BlobFUSE2
wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb
dpkg -i packages-microsoft-prod.deb
apt-get update && apt-get install blobfuse2 -y

# 2. Create the BlobFUSE2 Configuration
mkdir -p /home/azureuser/project
cat << 'EOF' > /home/azureuser/project/blob_config.yaml
allow-other: true
logging:
  type: syslog
  level: log_error
components:
  - libfuse
  - file_cache
  - attr_cache
  - azstorage
libfuse:
  attribute-timeout-sec: 120
  entry-timeout-sec: 120
  negative-entry-timeout-sec: 240
file_cache:
  path: /mnt/blob_cache
  timeout-sec: 120
  max-size-mb: 4096
azstorage:
  type: block
  account-name: stmlspotminimal123
  container-name: artifacts
  mode: msi
EOF

# 3. Setup Mount and Cache Folders
mkdir -p /mnt/blob_cache
mkdir -p /home/azureuser/project/data
chown -R azureuser:azureuser /home/azureuser/project

# 4. Mount the Blob Storage container using BlobFUSE2
blobfuse2 mount /home/azureuser/project/data --config-file=/home/azureuser/project/blob_config.yaml

# 5. Install Miniconda & Setup Environment
mkdir -p /opt/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda3/miniconda.sh
bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3
export PATH="/opt/miniconda3/bin:$PATH"

# 6. Clone your code and install requirements
git clone <YOUR_REPO_URL> /home/azureuser/project/src
cd /home/azureuser/project/src
conda create -n ml_env python=3.10 -y
source /opt/miniconda3/bin/activate ml_env

# Install your requirements if they exist
[ -f requirements.txt ] && pip install -r requirements.txt

# 7. Submit training job in the background
nohup python -u train.py --data_dir /home/azureuser/project/data > /home/azureuser/project/training.log 2>&1 &
