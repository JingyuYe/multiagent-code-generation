.PHONY: oof clean status tail kill

# Run the parallel experiment in the background
oof:
	@echo "🚀 Starting Parallel Experiment in the background..."
	@mkdir -p data
	@nohup python3 scripts/run_all_experiments.py > data/test_run.log 2>&1 &
	@echo "✅ Experiment started. Logs are being saved to data/test_run.log"
	@echo "📊 Monitor progress with: make tail"

# Check if the experiment is still running
status:
	@pgrep -af scripts/run_all_experiments.py || echo "❌ No experiment currently running."

# Live monitor logs
tail:
	@tail -f data/test_run.log

# Stop any running experiments
kill:
	@echo "🛑 Killing all running experiment processes..."
	@pkill -f scripts/run_all_experiments.py || echo "No processes found."

# Clear all data results and logs
clean:
	@echo "🧹 Cleaning up data directory..."
	rm -f data/*.log data/*.jsonl data/*.csv
