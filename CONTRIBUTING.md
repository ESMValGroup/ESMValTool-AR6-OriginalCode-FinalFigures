# IPCC AR6 diagnostics in the ESMValTool

The ESMValTool code for IPCC AR6 will be released in two steps:
 
  - **Step 1:** The original code that was used to produce the final IPCC figures will be collected and stored 
  - **Step 2:** The recipes and diagnostics will be updated and merged into the public ESMValTool version 

This needs to be done by the author of the diagnostics and supported by at least one person from the technical ESMValTool development team who is also a Contributing Author on one of the chapters. 

**Coordinators and Contact for questions:**

*General inquiries:*
- Lisa Bock (<lisa.bock@dlr.de>)
- Rémi Kazeroni (<Remi.Kazeroni@dlr.de>)

*Technical support (`git`, `conda`, `pip`-related questions):*
- Valeriu Predoi (<valeriu.predoi@ncas.ac.uk>)

## Step 1: Documentation of original code (basis for Zenodo citations (TSU will do this step))

- **Upload ESMValTool code that you used to create the final IPCC figures**

  In order to ensure that the figures can be reproduced, the original code used to produce the final IPCC AR6 figures will be collected. If you have used multiple branches to create your figures, do the following steps for each branch. Do not merge any branches at this point.

  - Please check that you are satisfied with your code’s comments and documentation. Do not change any active code at this stage, but you can add comments and documentation. The code and recipe header should fully describe the figures it produces and include their IPCC figure number. Please contact Lisa (email: <lisa.bock@dlr.de>) if the figure numbers are not known.
  - Make sure all relevant code is committed to your local git repository. You can check the status of the local repository with:
    ```
    git status
    ```
  - If any necessary files are missing, please add and commit them with:
    ```
    git add <pathtofilename>
    git commit -m "some message"
    ```
    **NOTE:** it is best you committed each file separateley and not using a `*` wildcard since it's useful to have each commit's hash ID and message logged; also when adding to the git repository using `git add` make sure you are not adding unnecessary files like temporary or log files (try not to use `git add *` as much as possible).
  - Make a full record of your current branch and commit details. Use the command:
    ```
    git log -n 1
    ```
    Take a note of the output, specifically the commit hash ID, which will be a long number and look something like: `215095e2e338525be0baeeebdf66bfbb304e7270`. This commit hash ID will allow you to fully restore your code in the case of an unforeseen error (but this can only be done before the branch gets merged and deleted!). If the branch has been merged, then we’ll have to track the issue from the merge commit hash ID (see below)
  - Upload your final code in your branch(es) on the [ESMValTool-AR6-OriginalCode-FinalFigures](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures) repository:
    ```
    git remote add origin2 https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures.git
    git push origin2 <your branch>
    ```
  Please note that this should really document the original code used to create the final figures without ANY changes

- **Upload ESMValCore developments**

  Additionally to the ESMValTool code, the ESMValCore version and further changes have to be saved. This only needs to be done if  changes have been made to the ESMValCore.
  - Use `git status` in your local copy of ESMValCore to check whether you have made any changes to it
  - Make sure all relevant code is committed to your local git repository. If any files added by you are missing from git's version control, please add and commit them with:
    ```
    git add <pathtofilename>
    git commit -m "some message"
    ```
    **NOTE:** it is best you committed each file separateley and not using a `*` wildcard since it's useful to have each commit's hash ID and message logged; also when adding to the git repository using `git add` make sure you are not adding unnecessary files like temporary or log files (try not to use `git add *` as much as possible).
  - Make a full record of your current branch and commit details. Use the command:
    ```
    git log –n 1
    ```
    Take a note of the output, specifically the commit hash ID, which will be a long number and look something like `215095e2e338525be0baeeebdf66bfbb304e7270`. This commit ID will allow you to fully restore your code in the case of some kind of unforeseen error.
  - Upload your final code in your branch(es) on the [ESMValCore-AR6-OriginalCode-FinalFigures](https://github.com/ESMValGroup/ESMValCore-AR6-OriginalCode-FinalFigures) repository:
    ```
    git remote add origin2 https://github.com/ESMValGroup/ESMValCore-AR6-OriginalCode-FinalFigures.git
    git push origin2 <your branch> 
    ```
  Please note that this should really document the original code used to create the final figures without ANY changes!

- **Save conda environment**

  Save your conda environment: follow these steps to take a snapshot of your full dependencies environment managed by `conda`; here `<environment>` is the name of the conda environment you used for the analysis (most probably `esmvaltool`), please change it with whatever you named your main environment.
  ```
  conda activate <environment>  # activate the environment you used to run the IPCC stuff
  ```
  
  Make a local copy of the Final figures repository:
  ```
  git clone https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures.git
  cd ESMValTool-AR6-OriginalCode-FinalFigures
  
  ```
  
  Create a new branch where you will put information about <your branch> that was just pushed to this repository:
  ```
  git checkout -b <info_on_your_branch>
 
  ```
  
  Make sure you are on your newly created branch and it is up to date
  ```
  git status
  ```
  Name your new envinoment:
  ```
  NAME='my_environment_name'
  ```
  Export your full environment:
  ```
  conda env export > IPCC_environments/$NAME_conda_environment.yml  # export full environment specs
  echo "# conda version:" >> IPCC_environments/$NAME_conda_environment.yml 2>&1  # add a field that records conda version
  conda -V >> IPCC_environments/$NAME_conda_environment.yml && sed -i '$ s/^/# "conda -V" /' IPCC_environments/$NAME_conda_environment.yml # record conda version
  git add IPCC_environments/$NAME_conda_environment.yml  # add your new file to git control
  git commit -m "added environment record for $NAME"
  ```

- **Save pip environment**

  Save your pip dependencies environment: follow these steps to take a snapshot of your full dependencies environment managed by `pip`:
  ```
  pip freeze > IPCC_environments/$NAME_pip_environment.txt
  git add IPCC_environments/$NAME_pip_environment.txt
  git commit IPCC_environments/$NAME_pip_environment.txt -m "added pip environment for $NAME"
  ```

- **Write `README`**

  - Create one `README` file for each figure produced with your ESMValTool branch. Please use this [README](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_README_files/README.md) as a template and do not include several figures in a given README. The README file should be named as: `IPCC_README_files/README_Chap_X_Fig_Y.md` where X refers to the chapter number and Y to the figure number.
    ```
    cd ESMValTool-AR6-OriginalCode-FinalFigures/IPCC_README_files
    # create here text a file with your favorite text editor: vim, gedit, etc
    ```
    Add to text file [see template](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_README_files/README.md):
      - Figure number(s)
      - Author list
      - Branch(es) in the repositories ESMValTool-AR6-OriginalCode-FinalFigures and ESMValCore-AR6-OriginalCode-FinalFigures
      - Recipe(s) and diagnostic(s) used 
      - Were any automated recipe generations tools used?
      - Software versions, name of environment file (see 3.), other software packages,…
      - Machine used (e.g. Mistral, Jasmin, ...)
      - Further instructions
      - List/Description of auxiliary data used to produce the IPCC AR6 figures
      - Publication sources
      - ESMValTool instructions to reproduce the same figure

  - Push the environment and README files to the ESMValTool-AR6-OriginalCode-FinalFigures repository
    ```
      git add IPCC_README_files/README_Chap_X_Fig_Y.md # add a readme file for every figure of your ESMValTool-AR6 branch
      git commit -m "added README for Fig Y of Chap X"
      git push --set-upstream origin <info_on_your_ipcc_branch> # replace `<info_on_your_ipcc_branch>` by the branch named used to create these files.
    ```
  - Open a new issue in ESMValTool-AR6-OriginalCode-FinalFigures's [GitHub issues](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/issues). In your new issue, please include the figure numbers for the `README` files created with the instructions above. Once created, please assign your issue to `valeriupredoi` or `remi-kazeroni` using the **Assignee** tab on the right. Take a note of your issue number, you will need it later. 
  - Open a new draft pull request in ESMValCore's [Pull Requests](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/pulls). When creating a pull request, the base branch should be `main` and the compare branch should be your branch name (e.g. `<info_on_your_ipcc_branch>`). Please make sure that you link to your issue number in the pull request description. Once created, please assign your pull request to `valeriupredoi` or `remi-kazeroni` using the **Assignee** tab on the right.

- **Create Zenodo citation**

  TSU will create the Zenodo citation using the information in the README file


## Step 2: Merge the code into the public version of the ESMValTool

- **Merge ESMValCore developments**

  All ESMValCore developments could be merged from now on in the public [ESMValCore](https://github.com/ESMValGroup/ESMValCore) repository. Ideally all IPCC AR6 developments are part of the next release in June 2021 (code freeze on June 7, 2021). This only needs to be done if changes have been made to the ESMValCore code by you:
  - Create your own branch(es) in the public ESMValCore repository
    ```
    git clone https://github.com/ESMValGroup/ESMValCore.git  # get the gitball
    cd ESMValCore
    git checkout -b my_branch  # create your branch; rename my_branch with a valid name
    ```
  - Upload your final code to this branch (or to multiple branches, if needed):
    ```
    # first copy over all the code you need to commit
    # be aware of paths e.g. recipes, diagnostics etc
    git status  # here you'll see all your changes that have not been commited
    # add new files, one by one, if any 
    git add <new_file>  # add all the new files that you created and need to add to git version control
    git commit <new_file> -m "commit message"  # commit new files; write a relevant commit message
    # please don't use git commit * -m "commit message", we want to keep track of individual commits
    ```
  - Open a new issue in ESMValCore's [GitHub issues](https://github.com/ESMValGroup/ESMValCore/issues). In your new issue, please describe your code, the figure it produces, and also include the contents of your `README` from above. Once created, please assign your issue to `valeriupredoi` and `remi-kazeroni` using the **Assignees** tab on the right. Take a note of your issue number, you will need it later. 
  - With the help of your partner from the ESMValTool technical development team, ensure that your code complies with the coding standards required by ESMValTool.
  - Open a new draft pull request in ESMValCore's [Pull Requests](https://github.com/ESMValGroup/ESMValCore/pulls). When creating a pull request, the base branch should be `main` and the compare branch should be your branch name. Please make sure that you link to your issue number in the pull request description. Once created, please assign your pull request to `valeriupredoi` and `remi-kazeroni` using the **Assignees** tab on the right. Please use the **Labels** tab on the right hand side to assign the label **ipcc** to your pull request. Once you have created the pull request, GitHub will automatically test your code for compliance with ESMValTool standards and you are expected to fix any non-compliances before your pull request can be completed. 
  - Find [here](https://docs.esmvaltool.org/projects/esmvalcore/en/latest/contributing.html#checklist-for-pull-requests) the checklist for pull requests in the documentation. 
  - Once your code passes automated testing, the ESMValTool team will review the pull request. You may receive some instructions on what needs to be changed before your code can be merged. Please work with your ESMValTool reviewer and discuss and address any further comments that they raise. Note that this process may take a while to run through. It is possible that the underlying ESVMalTool code will change and that you may need to bring in recent changes from the `main`. Please ask for help with this if needed. 
  - When your code passes both the automated and human review, your code is ready to be merged – congratulations!

- **Prepare ESMValTool code**

  The code will not be merged in the `main` branch of the ESMValTool-AR6 repository but will be prepared for a merge in the `main` branch of the public ESMValTool repository. It could then be merged after the report is published (beginning of August).
  - Open separate branches for each recipe in the repository ESMValTool-AR6
    ```
    cd ESMValTool-AR6
    git checkout -b <your_branch>
    ```
  - Update your code to the recent ESMValTool version by switching the default remote branch from `master` to `main` (if not done before and if you are working with a local copy of the ESMValTool repository): 
    ```
    git branch -m master main
    git fetch origin
    git branch -u origin/main main
    git remote set-head origin -a
    ```
    and merging the `main` in your branch:
    ```
    git checkout main
    git pull
    git checkout <your_branch>
    git merge main
    # maybe there are some conflicts to solve before merging is possible
    # please fix those if any, ask for help if you get stuck
    ```
  - Collect all recipes in the folder `recipes/ipccar6wg1ch3/`. Naming convention for recipes:
    ```
    recipe_eyring21ipcc_3-3-atmosphere.yml
    recipe_eyring21ipcc_3-3-atmosphere_Fig3-8.yml
    recipe_eyring21ipcc_3-3-atmosphere_CCB3-2.yml
    recipe_eyring21ipcc_3-4-cryosphere.yml
    recipe_eyring21ipcc_3-5-ocean.yml
    recipe_eyring21ipcc_3-6-biosphere.yml
    recipe_eyring21ipcc_3-7-modes.yml
    recipe_eyring21ipcc_3-8-synthesis_FigX.yml
    ```
  - With the help of your partner from the ESMValTool technical development team, ensure that your code complies with the coding standards required by ESMValTool.
  - Follow the same instructions to open an issue, then a pull request as for the ESMValCore case

## Support team

There will be support by ESMValTool core developers which are Contributing Authors:

**Chapter 3:** 
- Atmosphere: Lisa, Katja, Yu with support by Rémi
- Ocean: Lee, Liza with support by Valeriu
- Modes: Yu, Lisa with support by Rémi
- SeaIce + Extremes: Liza with support by TBD
- Biosphere: Tina
- Synthesis: Lisa

**Chapter 4:**
- Erich Fisher with support by Ruth and TBD

**Chapter 6:**
- Chaincy Kuo with support by Lisa, Rémi
…

