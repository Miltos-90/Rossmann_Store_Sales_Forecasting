# Rossmann Store Sales Forecasting

## Provision
- Azure Resource Group
- Azure Storage account (Blob container)
  - Use BlobFuse2 to mount the blob container as a local directory on the VM
- Spot VM
  - [VM Size](https://azure.microsoft.com/en-us/pricing/spot-advisor)
    - E16ps v5 -> 16 vCPUs, 128 GB RAM -> Spot instance cost 0.17 USD/hr
  - Azure Spot Discount: Set to Enable
  - Eviction Policy: Set to Delete
- Managed ID asigned to the VM so it can read/write to the storage account


I am trying to run a single Standard_E16ps_v5 Spot VM for a workload, which requires 16 cores.