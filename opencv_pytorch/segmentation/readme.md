# Project 4: Kaggle Competition - Semantic Segmentation submission

## File listing
- **project4-semantic-segmentation.ipynb**  
  Jupyter-lab notebook used locally to run this experiment
  I found the Kaggle environment very difficult to work in. The notebook was so long and unwieldy and I wanted to be able to break it up into more manageable .py files, but Kaggle did not make this easy, so it was simpler to work in jupyter-lab on my local machine.
- **README.md**  
  this file
- **trainer**  
  trainer code provided in the open-cv course. I don't think i made any changes to this code, except maybe some print statements here and there to help debug my own code.
- **segmentation**      
  this is the package that i created in order to break up the notebook into more manageable chunks. most of the code was recycled from code already provided on the course. The only classes i made significant changes to are as follows:
  - **SemSegDataset.py**  
    The main change to this class beyond slight tweaks to the provided code was to ensure that the images were padded such that their dimensions were always divisible by 32. This is necessary for passing through the linknet architecture as the image resolution is halved with each layer and then upscaled, so in order to end up with the same size image that you started with, the image dimensions must be divisible by two 5 times.
  - **Experiment.py**  
    The main change to this code was to include the option to load an existing model, rather than always train from scratch. This enabled me to train in short bursts until the required level of performance was reached.
  
## My Kaggle profile page
<https://www.kaggle.com/epipolarity/competitions?tab=active>
My submission score was 0.68444.
