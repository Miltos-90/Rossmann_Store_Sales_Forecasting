# Rossmann Store Sales Forecasting

## Submit VM job
 1. Upload: Upload the ML scripts and data package directly into the Azure Storage Container from the local machine
 2. Trigger: Tell Azure's control plane to run a starter wrapper script on the ML 
  
  
The VM wakes up, uses its assigned identity to download the ML package from the storage account, executes the training script, and drops the artifacts right back to the storage acount