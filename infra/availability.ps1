####################
# Check the availability of spot instances in the specified regions and size
####################

$regions = @("eastus", "northeurope")
$sizes   = '[{"sku":"Standard_E16as_v7"}]'  # '[{"sku":"Standard_E16as_v7"},{"sku":"Standard_E16s_v7"}]'
$count   = 1

foreach ($region in $regions) {
    Write-Host "Checking availability for region: $region"
    az compute-recommender spot-placement-score `
      --desired-locations $region `
      --desired-sizes $sizes `
      --desired-count $count `
      --location $region
}
