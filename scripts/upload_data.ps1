####################
# Upload local training data to Azure Blob Storage.
# This script uses AzCopy, which is a command-line utility designed for fast and reliable data transfer to and from Azure Storage.
# To get AzCopy run this command: 
# winget install -e --id Microsoft.Azure.AZCopy.10
####################

# 1. Define local paths and Azure constants
$localDataPath    = "C:\path\to\your\local\training_data_folder"  # <-- Update this path
$storageAccount   = "stmlspotminimal123"
$containerName    = "artifacts"
$destinationFolder = "data" # Creates a 'data/' directory inside the container

# 2. Ensure we are logged into Azure CLI locally
Write-Host "Checking Azure authentication..."
az account show > $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Not logged in. Running 'az login'..."
    az login
}

# 3. Create the container if it doesn't exist
Write-Host "Verifying blob container '$containerName' exists..."
# Using the --auth-mode login ensures it uses the current 'az login' credentials
az storage container create `
  --name $containerName `
  --account-name $storageAccount `
  --auth-mode login `
  --warning disable > $null

if ($LASTEXITCODE -ne 0) {
    Write-Error "CRITICAL: Could not verify or create the container '$containerName' on storage account '$storageAccount'. Aborting upload."
    exit $LASTEXITCODE
}
Write-Host "Container verified/created successfully."  # We can proceed with the upload.

# 4. Get a temporary login token for AzCopy from the Azure CLI session
Write-Host "Authenticating AzCopy..."
$env:AZCOPY_AUTO_LOGIN_TYPE = "AZCLI"

# 5. Run AzCopy to sync the files
Write-Host "Uploading data to Azure Blob Storage..."
# Using 'sync' ensures only new or modified files are uploaded if you run this multiple times
azcopy sync "$localDataPath" "https://$storageAccount.blob.core.windows.net/$containerName/$destinationFolder" --recursive

Write-Host "Upload complete!"
