source ~/.bashrc
conda activate sofia
python3 -c "from training import reset_csv; reset_csv()"
sbatch slurm/slurm.sh
watch -n 2 'squeue -u se0361'