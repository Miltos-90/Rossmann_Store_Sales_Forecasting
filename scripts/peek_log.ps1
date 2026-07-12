####################
# Check the last 50 lines of the training log on the specified VM
####################

az vm run-command invoke `
  --resource-group rg-ml-spot-minimal `
  --name vm-worker `
  --command-id RunShellScript `
  --scripts "tail -n 50 /home/azureuser/project/training.log"