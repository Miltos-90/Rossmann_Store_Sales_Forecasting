####################
# Uploads files from a local directory to an Azure storage container.
####################
$AccountName = "stmlws123"  # Storage account name
$Source = "./my_dataset" # Local directory containing the files to upload
$Destination = "localpath/data"  # Destination path in the storage container

az storage blob upload-batch `
  --account-name $AccountName `
  --destination $Destination `
  --source $Source `
  --auth-mode login