# Check availability of a vm size in a specific region using Azure CLI
# another location: northeurope
$sizes = '[{"sku":"Standard_E16ps_v5"}]'
az compute-recommender spot-placement-score `
  --desired-locations eastus `
  --desired-sizes $sizes `
  --desired-count 1 `
  --location "eastus"

