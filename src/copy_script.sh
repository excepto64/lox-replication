git clone https://github.com/excepto64/lox-replication.git
cd lox-replication

nano .env

sbatch ./src/run.sh

src/runs.sh 2>&1 | tee run.log

rm -rf *.pdf *.pt model logs wandb checkpoint