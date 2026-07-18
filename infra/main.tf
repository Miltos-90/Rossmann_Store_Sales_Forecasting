######### REQUIREMENTS #########
# Azure CLI: winget install -e --id Microsoft.AzureCLI
# Azure ML CLI: az extension add --name ml

######### CONSTANTS & PROVIDERS #########
locals {
  resource_group_location = "eastus"
  output_file_path        = "${path.module}/infra_config.json"
  
  # See Azure ML availability here: https://azure.microsoft.com/en-us/pricing/details/machine-learning/
  # See Azure compute availability here: https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/
  # Names do not always match between the two pages. Azure ML has a limited set of VM types available for training.
  use_spot = false  # Low priority = Spot VMs, Dedicated = Standard VMs.

  virtual_machine = local.use_spot ? {
    instance_type = "STANDARD_E16A_V4" # 16 vCPUs, 128 GiB RAM, No storage
    vm_priority   = "LowPriority"
  } : {
    instance_type = "STANDARD_F2S_V2"  # 2 vCPUs, 4 GiB RAM, No storage
    vm_priority   = "Dedicated"
  }

  cluster_scale_settings = {
    min_node_count                       = 0
    max_node_count                       = 1
    scale_down_nodes_after_idle_duration = "PT2M"  # ISO 8601 duration format (2 minutes)
  }
}

terraform {
  required_version = ">= 1.0.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false  # Allow Terraform to delete the resource group even if it contains resources.
    }
    machine_learning {
      purge_soft_deleted_workspace_on_destroy = true  # Bypasses the 14-day soft delete, permanently purging the ML workspace instantly.
    }
  }
}

# 1. Resource Group
resource "azurerm_resource_group" "ml_rg" {
  name     = "rg-ml-workspace-rossmann"
  location = local.resource_group_location
}

######### AZURE ML STUDIO DEPENDENCIES #########

# 2. Storage Account (Used for Datastores, scripts, and logs)
resource "azurerm_storage_account" "ml_storage" {
  name                     = "stmlws1233"
  resource_group_name      = azurerm_resource_group.ml_rg.name
  location                 = local.resource_group_location
  account_tier             = "Standard"
  account_replication_type = "LRS"  # Locally Redundant Storage - the cheapest option for non-critical data
}

# 2.1 Storage Container (Used to store datasets like MNIST)
resource "azurerm_storage_container" "ml_data_container" {
  name                  = "datacontainer1233"
  storage_account_id    = azurerm_storage_account.ml_storage.id
  container_access_type = "private"   # Keeps your datasets secure and private
}

# 3. Key Vault (Used to securely store storage account keys and secrets)
data "azurerm_client_config" "current" {}

resource "azurerm_key_vault" "ml_kv" {
  name                        = "kv-ml-workspace-1233"
  location                    = local.resource_group_location
  resource_group_name         = azurerm_resource_group.ml_rg.name
  tenant_id                   = data.azurerm_client_config.current.tenant_id
  sku_name                    = "standard"
  purge_protection_enabled    = false
}

# 4. Application Insights & Log Analytics (Used for training log telemetry)
resource "azurerm_log_analytics_workspace" "ml_law" {
  name                = "law-ml-telemetry"
  location            = local.resource_group_location
  resource_group_name = azurerm_resource_group.ml_rg.name
  sku                 = "PerGB2018"  # PerGB2018: Pay-as-you-go pricing model
}

resource "azurerm_application_insights" "ml_appinsights" {
  name                = "appi-ml-telemetry"
  location            = local.resource_group_location
  resource_group_name = azurerm_resource_group.ml_rg.name
  workspace_id        = azurerm_log_analytics_workspace.ml_law.id
  application_type    = "web"
}

# 5. Azure Container Registry (Used to store custom Python environments/Docker images)
resource "azurerm_container_registry" "ml_acr" {
  name                = "crmlws1233"
  resource_group_name = azurerm_resource_group.ml_rg.name
  location            = local.resource_group_location
  sku                 = "Basic"
  admin_enabled       = true
}

######### AZURE ML WORKSPACE & COMPUTE CLUSTER #########

# 6. The Core Azure ML Workspace
resource "azurerm_machine_learning_workspace" "ml_ws" {
  name                    = "mlw-workspace"
  location                = local.resource_group_location
  resource_group_name     = azurerm_resource_group.ml_rg.name
  application_insights_id = azurerm_application_insights.ml_appinsights.id
  key_vault_id            = azurerm_key_vault.ml_kv.id
  storage_account_id           = azurerm_storage_account.ml_storage.id
  container_registry_id        = azurerm_container_registry.ml_acr.id
  storage_account_access_type  = "Identity"
  

  identity {
    type = "SystemAssigned"
  }
}

# 7. Compute Cluster
resource "azurerm_machine_learning_compute_cluster" "ml_cluster" {
  name                          = "ml-training-cluster"
  location                      = local.resource_group_location
  machine_learning_workspace_id = azurerm_machine_learning_workspace.ml_ws.id
  
  vm_size     = local.virtual_machine.instance_type
  vm_priority = local.virtual_machine.vm_priority

  scale_settings {
    min_node_count                       = local.cluster_scale_settings.min_node_count
    max_node_count                       = local.cluster_scale_settings.max_node_count
    scale_down_nodes_after_idle_duration = local.cluster_scale_settings.scale_down_nodes_after_idle_duration
  }

  identity {
    type = "SystemAssigned"
  }
}

######### RBAC ROLE ASSIGNMENTS #########
# NOTE: The following roles are automatically created by Azure ML and are NOT managed here:
#   - Storage Blob Data Contributor     → workspace managed identity on the linked storage account
#   - Storage File Data Privileged Contributor → workspace managed identity on the linked storage account
#   - AcrPull                           → compute cluster managed identity on the linked ACR

# 1. STORAGE: Allow Terraform caller (your current identity) to upload blobs
resource "azurerm_role_assignment" "caller_blob_contributor" {
  scope                = "${azurerm_storage_account.ml_storage.id}/blobServices/default/containers/${azurerm_storage_container.ml_data_container.name}"
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = data.azurerm_client_config.current.object_id
}

# 2. STORAGE: Allow AML compute managed identity to write job outputs and checkpoints
resource "azurerm_role_assignment" "compute_blob_contributor" {
  scope                = azurerm_storage_account.ml_storage.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_machine_learning_compute_cluster.ml_cluster.identity[0].principal_id
}

# 3. KEY VAULT: Allow your current identity (Terraform caller) to manage secrets
# NOTE: Not currently used by the Rossmann workload. Pre-provisioned for when external
#       data source credentials or API keys need to be written into Key Vault.
resource "azurerm_role_assignment" "caller_kv_secrets" {
  scope                = azurerm_key_vault.ml_kv.id
  role_definition_name = "Key Vault Secrets Officer"
  principal_id         = data.azurerm_client_config.current.object_id
}

# 4. KEY VAULT: Allow AML Compute Cluster to access secrets
# NOTE: Not currently used by the Rossmann workload. Pre-provisioned for when training
#       scripts need to read credentials or API keys from Key Vault at runtime.
resource "azurerm_role_assignment" "compute_kv_secrets" {
  scope                = azurerm_key_vault.ml_kv.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_machine_learning_compute_cluster.ml_cluster.identity[0].principal_id
}

# 5. STORAGE (FILES): Allow Terraform caller to upload notebooks/scripts via Azure ML Studio UI
# NOTE: Not required for programmatic job submission. Needed if prototyping directly
#       in the Studio notebook editor or uploading scripts through the Studio UI.
resource "azurerm_role_assignment" "caller_file_contributor" {
  scope                = azurerm_storage_account.ml_storage.id
  role_definition_name = "Storage File Data Privileged Contributor"
  principal_id         = data.azurerm_client_config.current.object_id
}

# 6. REGISTRY: Allow Workspace to push custom environment images to ACR
resource "azurerm_role_assignment" "workspace_acr_contributor" {
  scope                = azurerm_container_registry.ml_acr.id
  role_definition_name = "AcrPush"
  principal_id         = azurerm_machine_learning_workspace.ml_ws.identity[0].principal_id
}

# 7. AML WORKSPACE: Allow Terraform caller to manage assets, submit jobs, and use the Studio UI
resource "azurerm_role_assignment" "caller_ml_admin" {
  scope                = azurerm_machine_learning_workspace.ml_ws.id
  role_definition_name = "AzureML Data Scientist"
  principal_id         = data.azurerm_client_config.current.object_id
}

######### AML ASSETS #########

# Register the data container as an AML Datastore with identity-based access
resource "azurerm_machine_learning_datastore_blobstorage" "ml_datastore" {
  name                  = azurerm_storage_container.ml_data_container.name
  workspace_id          = azurerm_machine_learning_workspace.ml_ws.id
  storage_container_id  = azurerm_storage_container.ml_data_container.id
  service_data_auth_identity = "WorkspaceSystemAssignedIdentity"
}

######### OUTPUTS #########

# Output the Azure ML Workspace name and resource group for easy reference
resource "local_file" "ml_workspace_info" {
  filename = local.output_file_path
  content  = jsonencode({
    resource_group       = azurerm_resource_group.ml_rg.name
    workspace_name       = azurerm_machine_learning_workspace.ml_ws.name
    compute_name         = azurerm_machine_learning_compute_cluster.ml_cluster.name
    container_name       = azurerm_storage_container.ml_data_container.name
    storage_account_name = azurerm_storage_account.ml_storage.name
    datastore_name       = azurerm_machine_learning_datastore_blobstorage.ml_datastore.name
  })

  # Automatically deletes the file during 'terraform destroy'
  provisioner "local-exec" {
    when    = destroy
    command = "powershell -ExecutionPolicy Bypass -Command \"if (Test-Path '${self.filename}') { Remove-Item '${self.filename}' -Force }\""
  }
}