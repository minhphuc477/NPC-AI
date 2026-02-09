# Remove large files from Git history
# WARNING: This rewrites Git history!

# Files to remove (from error message)
git filter-repo --path 'outputs/adapter/checkpoint-75/optimizer.pt' --invert-paths --force
git filter-repo --path 'outputs/adapter_dpo/checkpoint-100/optimizer.pt' --invert-paths --force
git filter-repo --path 'data/MNIST/cifar-10-python.tar.gz' --invert-paths --force

# Remove entire directories with large files
git filter-repo --path 'outputs/' --invert-paths --force
git filter-repo --path 'data/MNIST/' --invert-paths --force
git filter-repo --path 'data/cifar-10-batches-py/' --invert-paths --force

Write-Host "Git history cleaned! Now force push with:"
Write-Host "git push -u origin feature/bdnsca-ci --force"
