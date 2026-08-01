Set-Location src # The VM sets the working directory to the root of the repo, so we need to change it to src before running the script
python -m main --input_dir ../artifacts --output_dir ../output
Set-Location .. # Change back to the root of the repo after running the script