# IPCC AR6 diagnostics in the ESMValTool

The ESMValTool code for IPCC AR6 will be released in two steps: 
  - 1. The original code that was used to produce the final IPCC figures will be collected and stored; 
  - 2. The recipes and diagnostics will be updated and merged into the public ESMValTool version 
This needs to be done by the author of the diagnostics and supported by at least one person from the technical ESMValTool development team who is also a Contributing Author on one of the chapters. 

Coordinators and Contact for questions: 
Lisa Bock (email: <lisa.bock@dlr.de>) and Rémi Kazeroni (email: <Remi.Kazeroni@dlr.de>) 

Step 1: Documentation of original code 
🡪 Basis for Zenodo citations (TSU will do)

1. Upload ESMValTool code that you used to create the final IPCC figures
In order to ensure that the figures can be reproduced, the original code used to produce the final IPCC AR6 figures will be collected. If you have used multiple branches to create your figures, do the following steps for each branch. Do not merge any branches at this point.
    • Please check that you are satisfied with your code’s comments and documentation. Do not change any active code at this stage, but you can add comments and documentation. The code and recipe header should fully describe the figures it produces and include their IPCC figure number. Please contact Lisa (email: lisa.bock@dlr.de) if the figure numbers are not known.
    • Make sure all relevant code is committed to your local git repository. You can check the status of the local repository with:
> git status
    • If any necessary files are missing, please add and commit them with:
> git add <pathtofilename>
> git commit -m “some message”
    • Make a full record of your current branch and commit details. Use the command:
> git log –n 1 
Take a note of the output, specifically the commit id, which will be a long number and look something like: 215095e2e338525be0baeeebdf66bfbb304e7270. This commit ID will allow you to fully restore your code in the case of an unforeseen error.This is not necessary because git log can be used at any time as long as the branch is still active, if the branch has been merged then we’ll have to track the issue via the merge commit
    • Upload your final code in your branch(es) on the ESMValTool-AR6-OriginalCode-FinalFigures repository
>git remote add origin2 https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures.git
> git push origin2 <your branch>
Please note that this should really document the original code used to create the final figures without ANY changes

2. Upload ESMValCore developments
Additionally, to the ESMValTool code the ESMValCore version and further changes have to be saved. This only needs to be done if changes have been made to the ESMValCore.
    • You can check whether you have made any changes to ESMValCore by using the following command:
> ?
    • Make sure all relevant code is committed to your local git repository. You can check the status of the local repository with:
> git status
    • If any files are missing, please add and commit them with:
> git add <pathtofilename>
> git commit -m “some message”
    • Make a full record of your current branch and commit details. Use the command:
> git log –n 1 
Take a note of the output, specifically the commit id, which will be a long number and look something like: 215095e2e338525be0baeeebdf66bfbb304e7270. This commit ID will allow you to fully restore your code in the case of some kind of unforeseen error.
    • Upload your final code in your branch(es) on the ESMValCore-AR6-OriginalCode-FinalFigures repository
> git remote add origin2 https://github.com/ESMValGroup/ESMValCore-AR6-OriginalCode-FinalFigures.git
> git push origin2 <your branch> 
    • Please note that this should really document the original code used to create the final figures without ANY changes

3. Save environment version
    • Save your environment
> conda activate esmvaltool  # activate the environment you used to run the IPCC stuff
> cd ESMValTool-AR6-OriginalCode-FinalFigures
> git status  # make sure you are on your newly created branch and is up to date
> mkdir IPCC_environments # needed right??
> conda env export > IPCC_environments/$NAME_environment.yml  # export full environment specs; replace $NAME with relevant run name
> echo "# conda version:" >> IPCC_environments/$NAME_environment.yml 2>&1  # add a field that records conda version
> conda -V >> IPCC_environments/$NAME_environment.yml && sed -i '$ s/^/# "conda -V" /' IPCC_environments/$NAME_environment.yml # record conda version
> git add IPCC_environments/$NAME_environment.yml  # add your new file to git control
> git commit IPCC_environments/$NAME_environment.yml -m "added environment record for $NAME"  # push to remote
    • Save your pip environment
> pip freeze > IPCC_environments/$NAME_pip.txt
> git add IPCC_environments/$NAME_pip.txt
> git commit IPCC_environments/$NAME_pip.txt -m “added pip environment for $NAME”

4. Write README
    • Create README file in your branch
> cd ESMValTool-AR6-OriginalCode-FinalFigures
> mkdir IPCC_readme ??
> cd IPCC_readme
# create here text file with your favorite text editor
    • Add to text file:
        ◦ Figure number(s)
        ◦ Branch(es) in the repositories ESMValTool-AR6-OriginalCode-FinalFigures and ESMValCore-AR6-OriginalCode-FinalFigures
        ◦ Recipe(s) and diagnostic(s) used 
        ◦ Were any automated recipe generations tools used?
        ◦ Software versions, name of environment file (see 3.), other software packages,…
        ◦ Further instructions
        ◦ Author list
        ◦ List/Description of auxiliary data used to produce the IPCC AR6 figures
        ◦ Publication sources
        ◦ Machine used (e.g. Mistral, Jasmin, ...)
        ◦ something missing??? Would it be useful to include the esmvaltool instructions to reproduce the same figure or would it be too much: (git checkout branch; conda env create …; esmvaltool run recipe_xxx.yml)

5. Create Zenodo citation
TSU could create with the informations in the README the Zenodo citation


Step 2: Merge the code into the public version of the ESMValTool

1. Merge ESMValCore developments
All ESMValCore developments could be merged from now on in the public ESMValCore repository. Ideally all IPCC AR6 developments are part of the next release in June 2021 (code freeze on June 7). This only needs to be done if changes have been made to the ESMValCore.
    • Create your own branch(es) in the public ESMValCore repository
> git clone https://github.com/ESMValGroup/ESMValCore.git  # get the gitball
> cd ESMValCore
> git checkout -b my_branch  # create your branch; rename my_branch with a valid name
    • Upload your final code to this branch (or to multiple branches)
# copy over all the code you need to commit
# be aware of paths e.g. recipes, diagnostics etc
> git status  # here you'll see all your changes that have not been commited
> git add new_file  # add all the new files that you created and need to add to git version control
> git commit new_file -m "commit message"  # commit new files; write a relevant commit message
# please don't use git commit * -m "commit message", we want to keep track of individual commits
    • Open a new issue using the following link (https://github.com/ESMValGroup/ESMValCore/issues). In your new issue, please describe your code, the figure it produces, and also include the contents of your README, above. 
        ◦ Once created, please assign your issue to the valeriupredoi, remi-kazeroni  github user accounts.
        ◦ Take a note of your issue number, you will need it later. 
    • With the help of your partner from the ESMValTool technical development team, ensure that you code complies with the coding standards required by ESMValTool:
    • Open a new draft pull request: (https://github.com/ESMValGroup/ESMValCore/pulls)
        ◦ When creating a pull request, the base branch should be “master” and the compare branch should be your branch name.
        ◦ Please make sure that you link to your issue number in the pull request description. 
        ◦ Once created, please assign your pull request to the valeriupredoi, remi-kazeroni  github user accounts.
        ◦ Please use the menu on the right hand side to assign the label ipcc to your pull request.
    • Once you have created the pull request, GitHub will automatically test your code for compliance with ESMValTool standards and you are expected to fix any non-compliances before your pull request can be completed. 
    • Find here the checklist for pull requests in the documentation
    • Once your code passes automated testing, the ESMValTool team with review the pull request. You may receive some instructions on what needs to be changed before your code can be merged. Please work with your ESMValTool reviewer and address any further comments that they raise.
    • Note that this process may take a while to run through. It is possible that the underlying ESVMalTool code will change and that you may need to bring in recent changes from the master. Please ask for help with this if needed. 
    • When your code passes both the automated and human review, your code is ready to be merged – congratulations!

2. Prepare ESMValTool code
The code will not be merged in the master of the ESMValTool-AR6 repository but will be prepared for a merge in the master of the public ESMValTool repository. It could then be merged after the report is published (beginning of August).
    • Open separate branches for each recipe in the repository ESMValTool-AR6
> cd ESMValTool-AR6
> git checkout -b <your_branch>
    • Update your code to the recent ESMValTool version by merging the master in your branch
> git checkout master
> git pull
> git checkout <your_branch>
> git merge master
#maybe there are some conflicts to solve before merging is possible
    • Collect all recipes in the folder recipes/ipccar6wg1ch3/
Naming convention for recipes:
recipe_eyring21ipcc_3-3-atmosphere.yml
recipe_eyring21ipcc_3-3-atmosphere_Fig3-8.yml
recipe_eyring21ipcc_3-3-atmosphere_CCB3-2.yml
recipe_eyring21ipcc_3-4-cryosphere.yml
recipe_eyring21ipcc_3-5-ocean.yml
recipe_eyring21ipcc_3-6-biosphere.yml
recipe_eyring21ipcc_3-7-modes.yml
recipe_eyring21ipcc_3-8-synthesis_FigX.yml
    • With the help of your partner from the ESMValTool technical development team, ensure that you code complies with the coding standards required by ESMValTool:
    • Open a new draft pull request: (https://github.com/ESMValGroup/ESMValTool-AR6/pulls)
        ◦ When creating a pull request, the base branch should be “master” and the compare branch should be your branch name.
        ◦ Once created, please assign your pull request to the valeriupredoi, remi-kazeroni  github user accounts.
        ◦ Please use the menu on the right hand side to assign the label ipcc to your pull request.
    • Once you have created the pull request, GitHub will automatically test your code for compliance with ESMValTool standards and you are expected to fix any non-compliances before your pull request can be completed. 
    • Find here the checklist for pull requests in the documentation
    • Once your code passes automated testing, the ESMValTool team with review the pull request. You may receive some instructions on what needs to be changed before your code can be merged. Please work with your ESMValTool reviewer and address any further comments that they raise.
    • Note that this process may take a while to run through. It is possible that the underlying ESVMalTool code will change and that you may need to bring in recent changes from the master. Please ask for help with this if needed. 
    • When your code passes both the automated and human review, your code is ready to be merged – congratulations!
    • 
    • 

3. Merging of ESMValTool code after report is published
    • Transfer of pull requests from the private ESMValTool-AR6 repository in the public ESMValTool repository
    • Final review of pull requests and merging
---------------------------------------------------------------------------------------------------------------
There will be support by ESMValTool core developers which are Contributing Authors:
Chapter 3: 
Atmosphere 🡪 Lisa, Katja, Yu with support by Rémi
Ocean 🡪 Lee, Liza with support by Valeriu
Modes 🡪 Yu, Lisa with support by Rémi
SeaIce + Extremes 🡪 Liza with support by ?
Biosphere 🡪 Tina
Synthesis 🡪 Lisa
Chapter 4:
Erich Fisher with support by Ruth and ?
Chapter 6:
Chaincy Kuo with support by Lisa, Rémi
…

