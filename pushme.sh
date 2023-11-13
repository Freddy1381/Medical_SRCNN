#!/bin/bash

# Add all changes to the staging area
git add .

# Commit the changes with a default message
git commit -m "Update: Auto-commit by pushme.sh"

# Push the changes to the remote repository (assuming the default remote is 'origin' and the default branch is 'main')
git push origin master