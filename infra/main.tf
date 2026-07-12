######### CONSTANTS #########

locals {
  resource_group_location = "eastus"
  storage_account = {
    tier             = "Standard"
    replication_type = "LRS" # Cheapest option
  }

  use_spot_vm = true # true: Spot VM, false: Regular VM

  virtual_machine = local.use_spot_vm ? {
    # Spot VM configuration (pricing: https://azure.microsoft.com/en-us/pricing/spot-advisor/#pricing)
    instance_type   = "Standard_E16as_v7" # E16as v7: 16 vCPUs, 128 GiB RAM, No storage
    priority        = "Spot"
    eviction_policy = "Delete"
    max_bid_price   = -1
  } : {
    # Regular VM configuration (pricing: https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/#pricing)
    instance_type   = "Standard_B2ls_v2" # B2ls v2: 2 vCPUs, 4 GiB RAM, No storage
    priority        = "Regular"
  }

  # VM constants
  vm_ssh_key_path = "C:\\Users\\m_kal\\.ssh\\id_rsa.pub"
    # Run command `ssh-keygen -t rsa -b 4096` to generate a new SSH key pair if you don't have one already. 
    # Press Enter to accept the default file location. Press Enter again to skip setting a passphrase. 
    # Write down the location of the private key (usually C:\Users\<username>\.ssh\id_rsa) and keep it safe.
}

######### CONFIGURATION #########
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
      prevent_deletion_if_contains_resources = false
    }
  }
}

# Resource Group
resource "azurerm_resource_group" "ml_rg" {
  name     = "rg-ml-spot-minimal"
  location = local.resource_group_location
}


# Storage
resource "azurerm_storage_account" "ml_storage" {
  name                     = "stmlspotminimal123"
  resource_group_name      = azurerm_resource_group.ml_rg.name
  location                 = local.resource_group_location
  account_tier             = local.storage_account.tier
  account_replication_type = local.storage_account.replication_type
}


# RBAC
resource "azurerm_user_assigned_identity" "vm_identity" {
  name                = "id-vm-executor"
  resource_group_name = azurerm_resource_group.ml_rg.name
  location            = local.resource_group_location
}

resource "azurerm_role_assignment" "storage_contrib" {
  scope                = azurerm_storage_account.ml_storage.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.vm_identity.principal_id
}


# Networking
resource "azurerm_virtual_network" "ml_vnet" {
  name                = "vnet-ml"
  address_space       = ["10.0.0.0/16"]
  location            = local.resource_group_location
  resource_group_name = azurerm_resource_group.ml_rg.name
}

resource "azurerm_subnet" "ml_subnet" {
  name                 = "snet-vm"
  resource_group_name  = azurerm_resource_group.ml_rg.name
  virtual_network_name = azurerm_virtual_network.ml_vnet.name
  address_prefixes     = ["10.0.1.0/24"]
}

resource "azurerm_network_interface" "ml_nic" {
  name                = "nic-vm"
  location            = local.resource_group_location
  resource_group_name = azurerm_resource_group.ml_rg.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.ml_subnet.id
    private_ip_address_allocation = "Dynamic"
  }
}


# Virtual Machine
resource "azurerm_linux_virtual_machine" "spot_vm" {
  name                = "vm-worker"
  admin_username      = "azureuser"
  resource_group_name = azurerm_resource_group.ml_rg.name
  location            = local.resource_group_location
  size                = local.virtual_machine.instance_type
  priority            = local.virtual_machine.priority
  eviction_policy     = local.virtual_machine.eviction_policy # Deletes the OS disk upon eviction to stop billing entirely
  max_bid_price       = local.virtual_machine.max_bid_price   # Default to capacity-only eviction

  admin_ssh_key {
    username   = "azureuser"
    public_key = file(local.vm_ssh_key_path)
  }

  network_interface_ids = [
    azurerm_network_interface.ml_nic.id,
  ]

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts"
    version   = "latest"
  }

  # Attach the Managed Identity to this specific VM
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.vm_identity.id]
  }
}