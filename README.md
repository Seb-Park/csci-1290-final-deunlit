# csci-1290-final-deunlit
Making shadowed things not shadowed.

# How to run
Make sure you have activated the CS-1290 conda environment. `cd` into the `code` folder and simply run `python main.py`. The image to deshadow and its corresponding mask can be specified in `main.py`. Additionally, the number of iterations can be specified in `main.py`.

The images at each iteration will be saved to `results`, and the final post processed images will also be saved there as well. 

At the end of the `n` iterations, the code will display the results to `pyplot`. If you go through all of these and close each window, all the final images should save properly.