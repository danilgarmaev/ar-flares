#!/bin/bash
# Quick training starter script
# Usage: ./start_training.sh [optional_script_name]

SCRIPT=${1:-train_balanced_convnext.py}

echo "Starting training in tmux session..."
echo "Script: $SCRIPT"
echo ""

# Create a new tmux session named 'training' and run the script
tmux new-session -d -s training "cd /teamspace/studios/this_studio/AR-flares/classifier_NN/refactored && python $SCRIPT"

sleep 2

echo "✓ Training started in tmux session 'training'"
echo ""
echo "Commands:"
echo "  tmux attach -t training    # Attach to see live progress"
echo "  Ctrl+B then D              # Detach (keeps running)"
echo "  tmux ls                    # List sessions"
echo "  tmux kill-session -t training  # Stop training"
echo ""
echo "To monitor without attaching:"
echo "  watch -n 2 'tmux capture-pane -t training -p | tail -5'"
