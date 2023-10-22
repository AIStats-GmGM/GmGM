# Build fortran files using f2py

cd python-frontend

# Loop through all files in ../backend
for file in ../backend/*.f90; do
    # Get the filename without the extension
    filename=$(basename -- "$file")
    filename="${filename%.*}"

    # Convert to lowercase
    filename=$(echo $filename | tr "[:upper:]" "[:lower:]")

    # Compile the file
    #echo $filename
    python -m numpy.f2py -c $file -m $filename --lower --quiet
done