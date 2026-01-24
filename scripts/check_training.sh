#!/bin/bash
# Monitor training progress

echo "========================================="
echo "Training Monitor"
echo "========================================="

# Check process
PROC=$(ps aux | grep "python train_balanced" | grep -v grep | grep -v SCREEN | tail -1)
if [ -z "$PROC" ]; then
    echo "❌ No training process found"
else
    echo "✓ Training is running:"
    echo "$PROC" | awk '{print "  PID: "$2", CPU: "$3"%, MEM: "$4"%"}'
fi

echo ""
echo "GPU Usage:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | \
    awk -F', ' '{print "  GPU: "$1", Memory: "$2" / "$3}'

echo ""
echo "Latest log (training_output.log):"
echo "-----------------------------------------"
tail -15 training_output.log 2>/dev/null || echo "No log file yet"

echo ""
echo "========================================="
echo "To attach to training: screen -r training"
echo "To detach: Press Ctrl+A then D"
echo "========================================="
