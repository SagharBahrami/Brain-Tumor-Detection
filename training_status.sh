# Training Status Monitor
echo '=== Training Status ==='
date
echo 'Running processes:'
ps aux | grep 'train_model.py' | grep -v grep | wc -l
echo 'Recent log activity:'
tail -5 full_training.log
echo 'Experiment directories:'
ls -1 experiments_logs/ | grep 2025-11-27 | wc -l
