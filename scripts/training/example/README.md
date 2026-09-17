# Task template

Copy this directory to `scripts/training/<task>/`, edit `paths.env` and the
model script, then inspect it with `bash scripts/training/<task>/pi05.sh plan`.
The new task directory is ignored by Git automatically. This example is tracked
and uses placeholder dataset/task names; prepare real data before training.

Keep scheduler commands in `.local/training/`. When using a remote checkout,
copy the selected task directory as well as the shared scripts and XPolicyLab
source: a Git checkout alone does not include your ignored task settings.
