####################
# Run the training job on the VM
####################

az vm run-command invoke `
  --resource-group rg-ml-spot-minimal `
  --name vm-worker `
  --command-id RunShellScript `
  --scripts @setup_and_train.sh