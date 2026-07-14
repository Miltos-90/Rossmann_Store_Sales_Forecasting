######### CONSTANTS & PROVIDERS #########
locals {
  resource_group_location = "eastus"
  
  use_low_priority = true  # Low priority = Spot VMs, Dedicated = Standard VMs

  virtual_machine = local.use_low_priority ? {
    instance_type = "Standard_E16as_v7" # # E16as v7: 16 vCPUs, 128 GiB RAM, No storage
    vm_priority   = "LowPriority"
  } : {
    instance_type = "Standard_B2ls_v2"  # B2ls v2: 2 vCPUs, 4 GiB RAM, No storage
    vm_priority   = "Dedicated"         
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
  features {}
}

# 1. Resource Group
resource "azurerm_resource_group" "ml_rg" {
  name     = "rg-ml-workspace"
  location = local.resource_group_location
}

######### FOUNDATIONAL DEPENDENCIES #########

# 2. Storage Account (Used for Datastores, scripts, and logs)
resource "azurerm_storage_account" "ml_storage" {
  name                     = "stmlws123"
  resource_group_name      = azurerm_resource_group.ml_rg.name
  location                 = local.resource_group_location
  account_tier             = "Standard"
  account_replication_type = "LRS"  # Locally Redundant Storage - the cheapest option for non-critical data
}

# 2.1 Storage Container (Used to store datasets like MNIST)
resource "azurerm_storage_container" "ml_data_container" {
  name                  = "data_container" # Target container name
  storage_account_id    = azurerm_storage_account.ml_storage.id
  container_access_type = "private"   # Keeps your datasets secure and private
}

# 3. Key Vault (Used to securely store storage account keys and secrets)
data "azurerm_client_config" "current" {}

resource "azurerm_key_vault" "ml_kv" {
  name                        = "kv-ml-workspace-123"
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
  name                = "crmlws123"
  resource_group_name = azurerm_resource_group.ml_rg.name
  location            = local.resource_group_location
  sku                 = "Standard"
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
  storage_account_id      = azurerm_storage_account.ml_storage.id
  container_registry_id   = azurerm_container_registry.ml_acr.id

  identity {
    type = "SystemAssigned"
  }
}

# 7. The Dynamic Compute Cluster
# Scalable compute for model training
resource "azurerm_machine_learning_compute_cluster" "ml_cluster" {
  name                          = "ml-training-cluster"
  location                      = local.resource_group_location
  machine_learning_workspace_id = azurerm_machine_learning_workspace.ml_ws.id
  
  vm_size     = local.virtual_machine.instance_type
  vm_priority = local.virtual_machine.vm_priority

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = 1
    scale_down_nodes_after_idle_duration = "PT2M" 
  }

  identity {
    type = "SystemAssigned"
  }
}