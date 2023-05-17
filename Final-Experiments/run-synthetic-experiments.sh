# PR Curves
echo "2-axis PR curves"
python Final-Experiments/synthetic-data-pr-curves.py -v 1
echo "3-axis PR curves"
python Final-Experiments/synthetic-data-tensor-pr-curves.py -v 1
echo "Shared axis PR curves"
python Final-Experiments/synthetic-data-shared-axis-pr-curves.py -v 1

# Runtime
echo "2-axis runtime"
python Final-Experiments/synthetic-data-runtime.py -v 1 -K 2
echo "3-axis runtime"
python Final-Experiments/synthetic-data-runtime.py -v 1 -K 3