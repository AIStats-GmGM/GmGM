echo "Linking Data"
Rscript Final-Experiments/lifelines-assortativity-create-map.r

echo "Running GmGM"
python Final-Experiments/lifelines-assortativity-run-gmgm.py

echo "Creating Plots"
Rscript Final-Experiments/lifelines-deep-plot-assortativity.r

echo "Done"